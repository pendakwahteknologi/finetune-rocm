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
  tip "Install Docker, then re-run phase 3."
  exit 1
fi

if [ -z "$HF_TOKEN" ]; then
  status_err "HF token is required for comparison."
  tip "Set it first:"
  out "    export HF_TOKEN=\"<your_hf_token>\""
  out "    export HUGGINGFACE_HUB_TOKEN=\"\$HF_TOKEN\""
  exit 1
fi

mkdir -p "$HF_CACHE_DIR" results

banner "Phase 3: Generate Comparison Report"
kv "Base model" "$BASE_MODEL"
kv "LoRA adapter" "$LORA_ADAPTER"
kv "Image" "$IMAGE_NAME"
kv "Timezone" "$CONTAINER_TZ"
echo ""

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  status_err "Pre-built image not found: $IMAGE_NAME"
  tip "Run ./scripts/start_finetuning_process.sh prelim first."
  exit 1
fi

step 1 1 "Running comparison and report generation in Docker"
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
import html
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

def esc(value):
    return html.escape(str(value), quote=True)


def percent(part, whole):
    if whole <= 0:
        return 0.0
    return round((part / whole) * 100.0, 1)


def avg(values):
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def render_pass_block(title, data):
    response = esc(data["response"])
    in_tok = data["input_tokens"]
    out_tok = data["output_tokens"]
    seconds = data["generation_time"]
    tps = data["tokens_per_sec"]
    return (
        "<div class=\"pass-block\">"
        f"<div class=\"pass-head\"><span>{esc(title)}</span><span class=\"mono\">"
        f"{out_tok} tok | {seconds}s | {tps} tok/s</span></div>"
        f"<pre>{response}</pre>"
        f"<div class=\"pass-foot mono\">input {in_tok} tok</div>"
        "</div>"
    )


meta = results["metadata"]
comparisons = results["comparisons"]
base_model_name = meta["base_model"]
lora_adapter_name = meta["lora_adapter"]
generated_at = meta["timestamp"]
total_prompts = len(comparisons)
diff_count = sum(1 for c in comparisons if c["different"])
same_count = total_prompts - diff_count
diff_pct = percent(diff_count, total_prompts)

base_consistent_count = sum(1 for c in comparisons if c["base_consistent"])
fine_consistent_count = sum(1 for c in comparisons if c["finetuned_consistent"])
base_consistency_pct = percent(base_consistent_count, total_prompts)
fine_consistency_pct = percent(fine_consistent_count, total_prompts)

base_tps = avg([c["base_pass1"]["tokens_per_sec"] for c in comparisons])
fine_tps = avg([c["finetuned_pass1"]["tokens_per_sec"] for c in comparisons])
base_out_tok = avg([c["base_pass1"]["output_tokens"] for c in comparisons])
fine_out_tok = avg([c["finetuned_pass1"]["output_tokens"] for c in comparisons])

metric_cards = []
metric_cards.append(
    "<article class=\"metric-card tone-primary\">"
    "<p class=\"metric-label\">Prompts Evaluated</p>"
    f"<p class=\"metric-value\">{total_prompts}</p>"
    "<p class=\"metric-note\">A B test prompts executed in two passes per model</p>"
    "</article>"
)
metric_cards.append(
    "<article class=\"metric-card tone-accent\">"
    "<p class=\"metric-label\">Changed Responses</p>"
    f"<p class=\"metric-value\">{diff_count} <span class=\"unit\">({diff_pct}%)</span></p>"
    f"<p class=\"metric-note\">{same_count} prompts stayed similar to base output</p>"
    "</article>"
)
metric_cards.append(
    "<article class=\"metric-card tone-neutral\">"
    "<p class=\"metric-label\">Consistency</p>"
    f"<p class=\"metric-value\">B {base_consistency_pct}% | FT {fine_consistency_pct}%</p>"
    "<p class=\"metric-note\">Higher means lower randomness between pass one and pass two</p>"
    "</article>"
)
metric_cards.append(
    "<article class=\"metric-card tone-neutral\">"
    "<p class=\"metric-label\">Average Pass One Throughput</p>"
    f"<p class=\"metric-value\">B {base_tps} | FT {fine_tps} <span class=\"unit\">tok/s</span></p>"
    f"<p class=\"metric-note\">Avg output tokens B {base_out_tok}, FT {fine_out_tok}</p>"
    "</article>"
)
metric_cards_html = "".join(metric_cards)

cards = []
for idx, c in enumerate(comparisons, start=1):
    cid = c["id"]
    prompt = esc(c["prompt"])
    prompt_index = esc(f"Prompt {cid}")
    changed = c["different"]
    changed_class = "changed" if changed else "similar"
    changed_label = "Changed after fine tuning" if changed else "Similar to base output"
    base_consistency_label = "Base consistent" if c["base_consistent"] else "Base variable"
    fine_consistency_label = "Fine tuned consistent" if c["finetuned_consistent"] else "Fine tuned variable"
    base_consistency_class = "consistent" if c["base_consistent"] else "variable"
    fine_consistency_class = "consistent" if c["finetuned_consistent"] else "variable"
    prompt_search = esc(c["prompt"].lower())

    base_pass_html = render_pass_block("Base pass one", c["base_pass1"]) + render_pass_block("Base pass two", c["base_pass2"])
    fine_pass_html = render_pass_block("Fine tuned pass one", c["finetuned_pass1"]) + render_pass_block("Fine tuned pass two", c["finetuned_pass2"])

    cards.append(
        f"<article class=\"result-card {changed_class}\" data-changed=\"{str(changed).lower()}\" "
        f"data-prompt=\"{prompt_search}\" style=\"--index:{idx};\">"
        "<header class=\"result-head\">"
        f"<span class=\"prompt-index\">{prompt_index}</span>"
        f"<h3>{prompt}</h3>"
        "<div class=\"pill-row\">"
        f"<span class=\"pill {changed_class}\">{esc(changed_label)}</span>"
        f"<span class=\"pill {base_consistency_class}\">{esc(base_consistency_label)}</span>"
        f"<span class=\"pill {fine_consistency_class}\">{esc(fine_consistency_label)}</span>"
        "</div>"
        "</header>"
        "<div class=\"model-grid\">"
        "<section class=\"model-panel base\">"
        "<h4>Base model</h4>"
        f"{base_pass_html}"
        "</section>"
        "<section class=\"model-panel tuned\">"
        "<h4>Fine tuned model</h4>"
        f"{fine_pass_html}"
        "</section>"
        "</div>"
        "</article>"
    )

cards_html = "".join(cards)

html_template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fine Tuning Comparison Report</title>
  <style>
    @import url("https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap");

    :root {
      --bg: #f6f8f8;
      --paper: #ffffff;
      --ink: #102a43;
      --muted: #486581;
      --teal: #0f766e;
      --teal-soft: #ccfbf1;
      --orange: #c2410c;
      --orange-soft: #ffedd5;
      --line: #d9e2ec;
      --shadow: 0 12px 30px rgba(16, 42, 67, 0.10);
      --radius-xl: 20px;
      --radius-lg: 14px;
      --radius-md: 10px;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: "Manrope", "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 0% 0%, #d1fae5 0, transparent 36%),
        radial-gradient(circle at 100% 0%, #ffedd5 0, transparent 34%),
        linear-gradient(180deg, #f8fbfa 0%, var(--bg) 100%);
      min-height: 100vh;
      padding: 28px 18px 36px;
    }

    .app {
      max-width: 1320px;
      margin: 0 auto;
      animation: pageIn 500ms ease;
    }

    .hero {
      background: linear-gradient(120deg, #0f766e 0%, #0e7490 55%, #115e59 100%);
      color: #f0fdfa;
      border-radius: var(--radius-xl);
      padding: 26px 28px;
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }

    .hero::after {
      content: "";
      position: absolute;
      inset: -40% -20% auto auto;
      width: 360px;
      height: 360px;
      border-radius: 999px;
      background: radial-gradient(circle, rgba(251, 191, 36, 0.23), rgba(251, 191, 36, 0));
      pointer-events: none;
    }

    .hero h1 {
      margin: 0;
      font-size: clamp(1.4rem, 2.3vw, 2.1rem);
      line-height: 1.2;
    }

    .meta-grid {
      margin-top: 16px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px 18px;
      font-size: 0.95rem;
    }

    .meta-item {
      background: rgba(255, 255, 255, 0.14);
      border: 1px solid rgba(255, 255, 255, 0.24);
      border-radius: 10px;
      padding: 10px 12px;
      backdrop-filter: blur(1.5px);
    }

    .meta-item strong {
      display: block;
      font-size: 0.75rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      opacity: 0.9;
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }

    .metric-card {
      border-radius: var(--radius-lg);
      padding: 14px 14px 12px;
      background: var(--paper);
      border: 1px solid var(--line);
      box-shadow: 0 5px 14px rgba(16, 42, 67, 0.07);
    }

    .metric-card.tone-primary {
      border-color: #7dd3fc;
      background: linear-gradient(180deg, #f0fdfa, #ecfeff);
    }

    .metric-card.tone-accent {
      border-color: #fdba74;
      background: linear-gradient(180deg, #fff7ed, #fff1e8);
    }

    .metric-label {
      margin: 0;
      font-weight: 700;
      font-size: 0.88rem;
      color: var(--muted);
    }

    .metric-value {
      margin: 8px 0 6px;
      font-size: clamp(1.05rem, 1.45vw, 1.35rem);
      font-weight: 800;
      color: var(--ink);
      line-height: 1.2;
    }

    .metric-note {
      margin: 0;
      font-size: 0.78rem;
      color: #5f7285;
    }

    .unit {
      font-size: 0.84rem;
      font-weight: 700;
      color: var(--muted);
    }

    .toolbar {
      margin-top: 16px;
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      box-shadow: 0 6px 18px rgba(16, 42, 67, 0.06);
      padding: 12px;
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px 12px;
      position: sticky;
      top: 10px;
      z-index: 20;
      backdrop-filter: blur(6px);
    }

    .filter-group {
      display: inline-flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .filter-btn {
      border: 1px solid #b8c7d5;
      background: white;
      color: #1f3b57;
      border-radius: 999px;
      padding: 8px 12px;
      font-family: inherit;
      font-size: 0.84rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 180ms ease;
    }

    .filter-btn:hover {
      transform: translateY(-1px);
      box-shadow: 0 5px 10px rgba(16, 42, 67, 0.08);
    }

    .filter-btn.active {
      border-color: #0f766e;
      background: #0f766e;
      color: #f0fdfa;
    }

    #search-input {
      margin-left: auto;
      border: 1px solid #b8c7d5;
      border-radius: 999px;
      padding: 8px 14px;
      min-width: 220px;
      font-size: 0.86rem;
      font-family: inherit;
    }

    #search-input:focus {
      outline: none;
      border-color: #0f766e;
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.16);
    }

    .visible-count {
      font-size: 0.82rem;
      color: var(--muted);
      font-weight: 700;
      margin-left: 6px;
      white-space: nowrap;
    }

    .results {
      margin-top: 14px;
      display: grid;
      gap: 12px;
    }

    .result-card {
      background: var(--paper);
      border: 1px solid var(--line);
      border-left-width: 6px;
      border-radius: var(--radius-lg);
      box-shadow: 0 7px 18px rgba(16, 42, 67, 0.07);
      overflow: hidden;
      animation: cardIn 420ms ease both;
      animation-delay: calc(var(--index) * 45ms);
    }

    .result-card.changed {
      border-left-color: var(--orange);
    }

    .result-card.similar {
      border-left-color: #15803d;
    }

    .result-head {
      padding: 14px 16px 8px;
      border-bottom: 1px solid #e7eef4;
      background: linear-gradient(180deg, rgba(255, 255, 255, 1), rgba(249, 252, 252, 1));
    }

    .result-head h3 {
      margin: 8px 0;
      font-size: 1.03rem;
      line-height: 1.35;
    }

    .prompt-index {
      display: inline-block;
      font-size: 0.74rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #486581;
      font-weight: 800;
    }

    .pill-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 2px;
    }

    .pill {
      border-radius: 999px;
      font-size: 0.72rem;
      font-weight: 800;
      letter-spacing: 0.02em;
      padding: 4px 10px;
      border: 1px solid transparent;
    }

    .pill.changed {
      background: var(--orange-soft);
      color: #9a3412;
      border-color: #fdba74;
    }

    .pill.similar {
      background: #dcfce7;
      color: #166534;
      border-color: #86efac;
    }

    .pill.consistent {
      background: #dbeafe;
      color: #1d4ed8;
      border-color: #93c5fd;
    }

    .pill.variable {
      background: #fee2e2;
      color: #b91c1c;
      border-color: #fca5a5;
    }

    .model-grid {
      padding: 12px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .model-panel {
      border: 1px solid #d9e2ec;
      border-radius: var(--radius-md);
      background: #fcfdff;
      padding: 10px;
    }

    .model-panel.base {
      border-top: 4px solid #0ea5a3;
    }

    .model-panel.tuned {
      border-top: 4px solid #ea580c;
    }

    .model-panel h4 {
      margin: 0 0 8px;
      font-size: 0.94rem;
    }

    .pass-block {
      border: 1px solid #d9e2ec;
      border-radius: 9px;
      padding: 8px 9px;
      margin-bottom: 8px;
      background: white;
    }

    .pass-block:last-child {
      margin-bottom: 0;
    }

    .pass-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      font-size: 0.77rem;
      font-weight: 700;
      margin-bottom: 7px;
      color: #334e68;
    }

    pre {
      margin: 0;
      white-space: pre-wrap;
      word-wrap: break-word;
      font-size: 0.86rem;
      line-height: 1.42;
      color: #102a43;
    }

    .mono {
      font-family: "IBM Plex Mono", "Consolas", monospace;
      font-size: 0.74rem;
    }

    .pass-foot {
      margin-top: 6px;
      color: #627d98;
    }

    .empty-state {
      display: none;
      border: 1px dashed #b8c7d5;
      border-radius: var(--radius-md);
      padding: 14px;
      text-align: center;
      color: var(--muted);
      background: #f8fbfd;
      margin-top: 10px;
    }

    @keyframes pageIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @keyframes cardIn {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 1020px) {
      .metric-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .model-grid {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 720px) {
      body {
        padding: 14px 10px 24px;
      }
      .hero {
        padding: 18px 16px;
      }
      .meta-grid {
        grid-template-columns: 1fr;
      }
      .metric-grid {
        grid-template-columns: 1fr;
      }
      .toolbar {
        position: static;
      }
      #search-input {
        width: 100%;
        margin-left: 0;
      }
    }
  </style>
</head>
<body>
  <main class="app">
    <section class="hero">
      <h1>LoRA Fine Tuning Comparison Report</h1>
      <div class="meta-grid">
        <div class="meta-item"><strong>Base model</strong><span>__BASE_MODEL__</span></div>
        <div class="meta-item"><strong>LoRA adapter</strong><span>__LORA_ADAPTER__</span></div>
        <div class="meta-item"><strong>Generated</strong><span>__GENERATED_AT__</span></div>
        <div class="meta-item"><strong>Passes per model</strong><span>2</span></div>
      </div>
    </section>

    <section class="metric-grid">
      __METRIC_CARDS__
    </section>

    <section class="toolbar">
      <div class="filter-group">
        <button class="filter-btn active" data-filter="all">All</button>
        <button class="filter-btn" data-filter="changed">Changed</button>
        <button class="filter-btn" data-filter="similar">Similar</button>
      </div>
      <span class="visible-count" id="visible-count"></span>
      <input id="search-input" type="search" placeholder="Search prompt text">
    </section>

    <section id="results" class="results">
      __RESULT_CARDS__
    </section>
    <div id="empty-state" class="empty-state">No prompts match the active filter.</div>
  </main>

  <script>
    (() => {
      const cards = Array.from(document.querySelectorAll(".result-card"));
      const buttons = Array.from(document.querySelectorAll(".filter-btn"));
      const searchInput = document.getElementById("search-input");
      const visibleCount = document.getElementById("visible-count");
      const emptyState = document.getElementById("empty-state");
      let activeFilter = "all";

      const updateView = () => {
        const query = searchInput.value.trim().toLowerCase();
        let shown = 0;
        cards.forEach((card) => {
          const isChanged = card.dataset.changed === "true";
          const matchFilter =
            activeFilter === "all" ||
            (activeFilter === "changed" && isChanged) ||
            (activeFilter === "similar" && !isChanged);
          const promptText = card.dataset.prompt || "";
          const matchSearch = query === "" || promptText.includes(query);
          const show = matchFilter && matchSearch;
          card.style.display = show ? "" : "none";
          if (show) {
            shown += 1;
          }
        });
        visibleCount.textContent = `${shown} visible`;
        emptyState.style.display = shown === 0 ? "block" : "none";
      };

      buttons.forEach((btn) => {
        btn.addEventListener("click", () => {
          activeFilter = btn.dataset.filter;
          buttons.forEach((other) => other.classList.remove("active"));
          btn.classList.add("active");
          updateView();
        });
      });

      searchInput.addEventListener("input", updateView);
      updateView();
    })();
  </script>
</body>
</html>
"""

html_out = (
    html_template
    .replace("__BASE_MODEL__", esc(base_model_name))
    .replace("__LORA_ADAPTER__", esc(lora_adapter_name))
    .replace("__GENERATED_AT__", esc(generated_at))
    .replace("__METRIC_CARDS__", metric_cards_html)
    .replace("__RESULT_CARDS__", cards_html)
)

OUT_HTML.write_text(html_out, encoding="utf-8")

print(f"Saved JSON: {OUT_JSON}")
print(f"Saved HTML: {OUT_HTML}")
PYEOF
  '

echo ""
status_ok "Comparison complete"
section "Artifacts"
kv "JSON" "results/comparison_results.json"
kv "HTML" "results/comparison_report.html"
tip "Open in browser: file://$(pwd)/results/comparison_report.html"
