#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/cli_theme.sh"

BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
LORA_ADAPTER="${LORA_ADAPTER:-models/lora_simple_rocm_out}"
IMAGE_NAME="${IMAGE_NAME:-finetune-rocm:ready}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
CONTAINER_TZ="${CONTAINER_TZ:-Asia/Kuala_Lumpur}"

TZ_DOCKER_ARGS=(-e "TZ=$CONTAINER_TZ")
if [ -f "/usr/share/zoneinfo/$CONTAINER_TZ" ]; then
  TZ_DOCKER_ARGS+=(-v "/usr/share/zoneinfo/$CONTAINER_TZ:/etc/localtime:ro")
elif [ -f /etc/localtime ]; then
  TZ_DOCKER_ARGS+=(-v /etc/localtime:/etc/localtime:ro)
fi
if [ -f /etc/timezone ]; then
  TZ_DOCKER_ARGS+=(-v /etc/timezone:/etc/timezone:ro)
fi

if ! command -v docker >/dev/null 2>&1; then
  status_err "Docker is required but not found in PATH."
  out "  Install Docker, then re-run phase 3."
  exit 1
fi

if [ -z "$HF_TOKEN" ]; then
  status_err "HF token is required for comparison."
  out "  Set it first:"
  out "    export HF_TOKEN=\"<your_hf_token>\""
  out "    export HUGGINGFACE_HUB_TOKEN=\"\$HF_TOKEN\""
  exit 1
fi

mkdir -p "$HF_CACHE_DIR" results

banner "Phase 3: Generate Comparison Report"
out "  ${WHITE}${BOLD}Base Model${RESET}  $BASE_MODEL"
out "  ${WHITE}${BOLD}LoRA Model${RESET}  $LORA_ADAPTER"
out "  ${WHITE}${BOLD}Image${RESET}       $IMAGE_NAME"
out "  ${WHITE}${BOLD}TZ${RESET}          $CONTAINER_TZ"
echo ""

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  status_err "Pre-built image not found: $IMAGE_NAME"
  out "  Run ./scripts/start_finetuning_process.sh prelim first."
  exit 1
fi

docker run --rm \
  "${TZ_DOCKER_ARGS[@]}" \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v "$(pwd)":/workspace \
  -v "$HF_CACHE_DIR":/root/.cache/huggingface \
  -w /workspace \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
  -e BASE_MODEL="$BASE_MODEL" \
  -e LORA_ADAPTER="$LORA_ADAPTER" \
  -e PYTHONUNBUFFERED=1 \
  "$IMAGE_NAME" bash -lc '
    missing=()
    for pkg in transformers peft accelerate; do
      python -c "import $pkg" >/dev/null 2>&1 || missing+=("$pkg")
    done
    if [ ${#missing[@]} -gt 0 ]; then
      pip install -q "${missing[@]}" --retries 10 --timeout 60
    fi

    python - <<"PYEOF"
import json
import os
import time
import gc
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.environ.get("BASE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
LORA_ADAPTER = os.environ.get("LORA_ADAPTER", "models/lora_simple_rocm_out")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
OUT_JSON = Path("results/comparison_results.json")
OUT_HTML = Path("results/comparison_report.html")

PROMPTS = [
    "Explain what Python is in simple terms.",
    "What is machine learning?",
    "How does a neural network work?",
    "What are the benefits of Docker?",
    "What is the difference between AI and ML?",
    "How does the internet work?",
    "What is a REST API?",
    "What is containerization?",
]

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7


def gen(model, tok, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    in_len = inputs.input_ids.shape[1]

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
        )
    dt = time.time() - t0

    resp = tok.decode(out[0][in_len:], skip_special_tokens=True).strip()
    out_tokens = int(out.shape[1] - in_len)
    return {
        "response": resp,
        "input_tokens": int(in_len),
        "output_tokens": out_tokens,
        "generation_time": round(dt, 2),
        "tokens_per_sec": round(out_tokens / dt, 2) if dt > 0 else 0,
    }


print("Loading base model...")
base_tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
if base_tok.pad_token is None:
    base_tok.pad_token = base_tok.eos_token
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN,
)

results = {
    "metadata": {
        "base_model": BASE_MODEL,
        "lora_adapter": LORA_ADAPTER,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_prompts": len(PROMPTS),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "passes_per_model": 2,
    },
    "comparisons": [],
}

base_runs = []
for i, prompt in enumerate(PROMPTS, start=1):
    print(f"[base {i}/{len(PROMPTS)}] {prompt}")
    b1 = gen(base_model, base_tok, prompt)
    b2 = gen(base_model, base_tok, prompt)
    base_runs.append((b1, b2))

print("Releasing base model from memory...")
del base_model
del base_tok
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Loading fine tuned model...")
ft_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN,
)
ft_tok = AutoTokenizer.from_pretrained(LORA_ADAPTER)
if ft_tok.pad_token is None:
    ft_tok.pad_token = ft_tok.eos_token
ft_model = PeftModel.from_pretrained(ft_base, LORA_ADAPTER)

for i, prompt in enumerate(PROMPTS, start=1):
    print(f"[fine tuned {i}/{len(PROMPTS)}] {prompt}")
    f1 = gen(ft_model, ft_tok, prompt)
    f2 = gen(ft_model, ft_tok, prompt)
    b1, b2 = base_runs[i - 1]

    results["comparisons"].append({
        "id": i,
        "prompt": prompt,
        "base_pass1": b1,
        "base_pass2": b2,
        "base_consistent": b1["response"] == b2["response"],
        "finetuned_pass1": f1,
        "finetuned_pass2": f2,
        "finetuned_consistent": f1["response"] == f2["response"],
        "different": b1["response"] != f1["response"],
    })

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

# Simple, clean HTML report
rows = []
for c in results["comparisons"]:
    cid = c["id"]
    prompt = c["prompt"]
    base_resp = c["base_pass1"]["response"]
    tuned_resp = c["finetuned_pass1"]["response"]
    changed = "YES" if c["different"] else "NO"
    rows.append(
        f"<tr>"
        f"<td>{cid}</td>"
        f"<td>{prompt}</td>"
        f"<td><pre>{base_resp}</pre></td>"
        f"<td><pre>{tuned_resp}</pre></td>"
        f"<td>{changed}</td>"
        f"</tr>"
    )

diff_count = sum(1 for c in results["comparisons"] if c["different"])
meta = results["metadata"]
base_model_name = meta["base_model"]
lora_adapter_name = meta["lora_adapter"]
generated_at = meta["timestamp"]
total_prompts = len(results["comparisons"])
rows_html = "".join(rows)

html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fine-Tuning Comparison Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f5f5f5; }}
    pre {{ white-space: pre-wrap; margin: 0; }}
  </style>
</head>
<body>
  <h1>Fine-Tuning Comparison Report</h1>
  <p><strong>Base model:</strong> {base_model_name}</p>
  <p><strong>LoRA adapter:</strong> {lora_adapter_name}</p>
  <p><strong>Generated:</strong> {generated_at}</p>
  <p><strong>Different responses:</strong> {diff_count}/{total_prompts}</p>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Prompt</th>
        <th>Base (pass 1)</th>
        <th>Fine-tuned (pass 1)</th>
        <th>Different?</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</body>
</html>
"""
OUT_HTML.write_text(html, encoding="utf-8")

print(f"Saved JSON: {OUT_JSON}")
print(f"Saved HTML: {OUT_HTML}")
PYEOF
  '

echo ""
status_ok "Comparison complete"
out "  Results saved to: results/comparison_results.json"
out "  HTML report: results/comparison_report.html"
out "  Open in browser: file://$(pwd)/results/comparison_report.html"
