#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/cli_theme.sh"

IMAGE_NAME="${IMAGE_NAME:-finetune-rocm:ready}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
BASE_DIR="/workspace"
OPENORCA_N="${OPENORCA_N:-5000}"
EVAL_N="${EVAL_N:-200}"
SEED="${SEED:-42}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant.}"

if ! command -v docker >/dev/null 2>&1; then
  status_err "Docker is required but not found in PATH."
  out "  Install Docker, then re-run phase 1."
  exit 1
fi

mkdir -p "$HF_CACHE_DIR"

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  status_err "Pre-built image not found: $IMAGE_NAME"
  out "  Run ./scripts/start_finetuning_process.sh prelim first."
  exit 1
fi

banner "Phase 1: Prepare Training Data"
out "  ${WHITE}${BOLD}Image${RESET}         $IMAGE_NAME"
out "  ${WHITE}${BOLD}OpenOrca Rows${RESET}  $OPENORCA_N"
out "  ${WHITE}${BOLD}Eval Rows${RESET}      $EVAL_N"
out "  ${WHITE}${BOLD}Seed${RESET}          $SEED"
echo ""

docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v "$(pwd)":/workspace \
  -v "$HF_CACHE_DIR":/root/.cache/huggingface \
  -w /workspace \
  -e BASE_DIR="$BASE_DIR" \
  -e OPENORCA_N="$OPENORCA_N" \
  -e EVAL_N="$EVAL_N" \
  -e SEED="$SEED" \
  -e SYSTEM_PROMPT="$SYSTEM_PROMPT" \
  "$IMAGE_NAME" python - <<'PYEOF'
import json
import os
import random
from pathlib import Path

from datasets import load_dataset


def clean_text(x):
    if x is None:
        return ""
    return str(x).strip()


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def looks_like_cot_trigger(text: str) -> bool:
    t = (text or "").lower()
    triggers = [
        "think step-by-step",
        "think step by step",
        "show your steps",
        "justify your steps",
        "explain your steps",
        "let's think step by step",
        "chain-of-thought",
        "show your reasoning",
    ]
    return any(x in t for x in triggers)


base_dir = Path(os.environ.get("BASE_DIR", "/workspace"))
openorca_n = int(os.environ.get("OPENORCA_N", "5000"))
eval_n = int(os.environ.get("EVAL_N", "200"))
seed = int(os.environ.get("SEED", "42"))
system_prompt = os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant.")

random.seed(seed)

data_dir = base_dir / "data"
logs_dir = base_dir / "logs"
data_dir.mkdir(parents=True, exist_ok=True)
logs_dir.mkdir(parents=True, exist_ok=True)

raw_out = data_dir / f"dolly15k_plus_openorca{openorca_n}.jsonl"
preview_out = data_dir / "preview_20_lines.jsonl"
cols_out = logs_dir / "openorca_columns.txt"
clean_out = data_dir / "train_clean.jsonl"
eval_out = data_dir / "eval_prompts.jsonl"
chatml_out = data_dir / "train_chatml.jsonl"
report_out = logs_dir / "clean_report.txt"

print("[1/3] Building merged dataset...")
dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
openorca = load_dataset("Open-Orca/OpenOrca", split="train")

cols_out.write_text("OpenOrca columns:\n" + ", ".join(openorca.column_names) + "\n", encoding="utf-8")

dolly_rows = []
for ex in dolly:
    dolly_rows.append({
        "instruction": clean_text(ex.get("instruction")),
        "input": clean_text(ex.get("context")),
        "output": clean_text(ex.get("response")),
        "source": "dolly15k",
    })

idxs = list(range(len(openorca)))
random.shuffle(idxs)
idxs = idxs[:openorca_n]
openorca_rows = []
for i in idxs:
    ex = openorca[i]
    question = ex.get("question") or ex.get("prompt") or ex.get("instruction") or ex.get("input") or ""
    answer = ex.get("response") or ex.get("completion") or ex.get("output") or ex.get("answer") or ""
    system = ex.get("system_prompt") or ex.get("system") or ex.get("systemPrompt") or ""
    row = {
        "instruction": clean_text(question),
        "input": clean_text(system),
        "output": clean_text(answer),
        "source": "openorca",
    }
    if row["instruction"] and row["output"]:
        openorca_rows.append(row)

all_rows = dolly_rows + openorca_rows[:openorca_n]
random.shuffle(all_rows)
write_jsonl(raw_out, all_rows)
write_jsonl(preview_out, all_rows[:20])
print(f"Saved raw dataset: {raw_out} ({len(all_rows)} rows)")

print("[2/3] Cleaning and splitting eval prompts...")
removed_empty = 0
removed_cot = 0
cleaned = []
for r in all_rows:
    instr = (r.get("instruction") or "").strip()
    inp = (r.get("input") or "").strip()
    out = (r.get("output") or "").strip()

    if not instr or not out:
        removed_empty += 1
        continue
    if looks_like_cot_trigger(instr) or looks_like_cot_trigger(inp):
        removed_cot += 1
        continue

    cleaned.append({
        "instruction": instr,
        "input": inp,
        "output": out,
        "source": r.get("source", ""),
    })

random.shuffle(cleaned)
eval_rows = cleaned[:eval_n]
train_rows = cleaned[eval_n:]

write_jsonl(clean_out, train_rows)
write_jsonl(eval_out, [{"instruction": r["instruction"], "input": r["input"], "source": r.get("source", "")} for r in eval_rows])

report = (
    f"Input rows: {len(all_rows)}\n"
    f"Removed empty: {removed_empty}\n"
    f"Removed chain-of-thought triggers: {removed_cot}\n"
    f"Clean train rows: {len(train_rows)}\n"
    f"Eval prompts: {len(eval_rows)}\n"
    f"Input file: {raw_out}\n"
)
report_out.write_text(report, encoding="utf-8")
print(f"Saved clean train: {clean_out}")
print(f"Saved eval prompts: {eval_out}")

print("[3/3] Converting clean train set to ChatML...")
chatml_rows = []
for r in read_jsonl(clean_out):
    instr = (r.get("instruction") or "").strip()
    inp = (r.get("input") or "").strip()
    out = (r.get("output") or "").strip()
    if not instr or not out:
        continue

    user = instr if not inp else f"{instr}\n\n{inp}"
    chatml_rows.append({
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user},
            {"role": "assistant", "content": out},
        ]
    })

write_jsonl(chatml_out, chatml_rows)
print(f"Saved ChatML train set: {chatml_out} ({len(chatml_rows)} rows)")
print("Phase 1 complete.")
PYEOF

if [ ! -f "data/train_chatml.jsonl" ]; then
  status_err "Expected output missing: data/train_chatml.jsonl"
  out "  Re-run prepare and ensure you are in the repository root."
  exit 1
fi

echo ""
status_ok "Data prep complete"
out "  Outputs:"
out "    - data/dolly15k_plus_openorca${OPENORCA_N}.jsonl"
out "    - data/train_clean.jsonl"
out "    - data/eval_prompts.jsonl"
out "    - data/train_chatml.jsonl"
