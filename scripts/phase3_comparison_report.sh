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
echo ""

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  status_err "Pre-built image not found: $IMAGE_NAME"
  out "  Run ./scripts/start_finetuning_process.sh prelim first."
  exit 1
fi

docker run --rm \
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
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN,
)

print("Loading fine-tuned model...")
ft_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    token=HF_TOKEN,
)
ft_base = ft_base.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
ft_tok = AutoTokenizer.from_pretrained(LORA_ADAPTER)
ft_model = PeftModel.from_pretrained(ft_base, LORA_ADAPTER)

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

for i, prompt in enumerate(PROMPTS, start=1):
    print(f"[{i}/{len(PROMPTS)}] {prompt}")
    b1 = gen(base_model, base_tok, prompt)
    b2 = gen(base_model, base_tok, prompt)
    f1 = gen(ft_model, ft_tok, prompt)
    f2 = gen(ft_model, ft_tok, prompt)

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
    rows.append(
        f"<tr>"
        f"<td>{c['id']}</td>"
        f"<td>{c['prompt']}</td>"
        f"<td><pre>{c['base_pass1']['response']}</pre></td>"
        f"<td><pre>{c['finetuned_pass1']['response']}</pre></td>"
        f"<td>{'YES' if c['different'] else 'NO'}</td>"
        f"</tr>"
    )

diff_count = sum(1 for c in results["comparisons"] if c["different"])
html = f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
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
  <p><strong>Base model:</strong> {results['metadata']['base_model']}</p>
  <p><strong>LoRA adapter:</strong> {results['metadata']['lora_adapter']}</p>
  <p><strong>Generated:</strong> {results['metadata']['timestamp']}</p>
  <p><strong>Different responses:</strong> {diff_count}/{len(results['comparisons'])}</p>
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
      {''.join(rows)}
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
