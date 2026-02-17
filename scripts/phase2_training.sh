#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/cli_theme.sh"

IMAGE_NAME="${IMAGE_NAME:-finetune-rocm:ready}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
CONTAINER_NAME="${CONTAINER_NAME:-rocm-training}"
TRAIN_LOG="${TRAIN_LOG:-results/training.log}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"

# Defaults (can be overridden by env)
MAX_TRAIN_ROWS_DEFAULT="${MAX_TRAIN_ROWS:-2000}"
BATCH_SIZE_DEFAULT="${BATCH_SIZE:-1}"
GRAD_ACCUM_DEFAULT="${GRAD_ACCUM:-4}"
USE_FP16_DEFAULT="${USE_FP16:-1}"
USE_BF16_DEFAULT="${USE_BF16:-0}"
DATALOADER_WORKERS_DEFAULT="${DATALOADER_WORKERS:-0}"
OUT_DIR_DEFAULT="${OUT_DIR:-/workspace/models/lora_simple_rocm_out}"

MODE="foreground"
FULL_RUN="0"

usage() {
  cat <<USAGE
Usage:
  ./scripts/phase2_training.sh [options]

Options:
  --background       Run training in background (detached, logs to results/training.log)
  --status           Show current background training status
  --stop             Stop background training container
  --full             Use full dataset defaults (MAX_TRAIN_ROWS=20011, output=lora_full_20k_rocm_out)
  -h, --help         Show this help

Examples:
  ./scripts/phase2_training.sh
  ./scripts/phase2_training.sh --full
  ./scripts/phase2_training.sh --background --full
  ./scripts/phase2_training.sh --status
  ./scripts/phase2_training.sh --stop
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --background) MODE="background" ;;
    --status) MODE="status" ;;
    --stop) MODE="stop" ;;
    --full) FULL_RUN="1" ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $arg"
      usage
      exit 1
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  status_err "Docker is required but not found in PATH."
  out "  Install Docker, then re-run phase 2."
  exit 1
fi

status() {
  banner "Phase 2: Training Status"

  if docker ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    status_ok "Training is running"
    docker ps --filter "name=^/${CONTAINER_NAME}$" --format "table {{.Names}}\t{{.Status}}\t{{.RunningFor}}"
    if [ -f "$TRAIN_LOG" ]; then
      section "Last 20 Log Lines"
      out "  ${DIM}$TRAIN_LOG${RESET}"
      tail -n 20 "$TRAIN_LOG"
    fi
    echo ""
    out "  Live logs: tail -f $TRAIN_LOG"
    out "  Stop: ./scripts/phase2_training.sh --stop"
    return 0
  fi

  status_warn "Training is not running"
  if [ -f "$TRAIN_LOG" ]; then
    if grep -q "Training Complete!" "$TRAIN_LOG"; then
      status_ok "Last run appears to have finished successfully"
    else
      status_warn "Last run may have stopped with errors"
    fi
    section "Last 30 Log Lines"
    out "  ${DIM}$TRAIN_LOG${RESET}"
    tail -n 30 "$TRAIN_LOG"
  else
    status_warn "No training log found yet: $TRAIN_LOG"
  fi
}

stop_training() {
  if docker ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    docker stop "$CONTAINER_NAME"
    status_ok "Stopped container: $CONTAINER_NAME"
  else
    status_warn "No running training container found: $CONTAINER_NAME"
  fi
}

if [ "$MODE" = "status" ]; then
  status
  exit 0
fi

if [ "$MODE" = "stop" ]; then
  stop_training
  exit 0
fi

if [ -z "$HF_TOKEN" ]; then
  status_err "HF token is required for training."
  out "  Set it first:"
  out "    export HF_TOKEN=\"<your_hf_token>\""
  out "    export HUGGINGFACE_HUB_TOKEN=\"\$HF_TOKEN\""
  exit 1
fi

mkdir -p "$HF_CACHE_DIR" results

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  status_err "Pre-built image not found: $IMAGE_NAME"
  out "  Run ./scripts/start_finetuning_process.sh prelim first."
  exit 1
fi

if [ ! -f "data/train_chatml.jsonl" ]; then
  status_err "Training dataset not found: data/train_chatml.jsonl"
  out "  Run: ./scripts/start_finetuning_process.sh prepare"
  out "  Then verify: ls -lh data/train_chatml.jsonl"
  exit 1
fi

MAX_TRAIN_ROWS_RUN="$MAX_TRAIN_ROWS_DEFAULT"
OUT_DIR_RUN="$OUT_DIR_DEFAULT"
if [ "$FULL_RUN" = "1" ]; then
  MAX_TRAIN_ROWS_RUN="${MAX_TRAIN_ROWS_FULL:-20011}"
  OUT_DIR_RUN="${OUT_DIR_FULL:-/workspace/models/lora_full_20k_rocm_out}"
fi

run_cmd=$(cat <<CMD
set -euo pipefail

# ROCm stability settings
export GPU_MAX_HW_QUEUES=2
export HSA_ENABLE_SDMA=0

# Training config
export USE_FP16=${USE_FP16_DEFAULT}
export USE_BF16=${USE_BF16_DEFAULT}
export BATCH_SIZE=${BATCH_SIZE_DEFAULT}
export GRAD_ACCUM=${GRAD_ACCUM_DEFAULT}
export MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS_RUN}
export DATALOADER_WORKERS=${DATALOADER_WORKERS_DEFAULT}
export BASE_DIR=/workspace
export DATA_DIR=/workspace/data
export OUT_DIR=${OUT_DIR_RUN}

echo 'Testing GPU...'
python -c 'import torch; print(f"GPU: {torch.cuda.get_device_name(0)}"); x=torch.randn(100,100,device="cuda"); print("GPU test OK")'

echo 'Starting training...'
python - <<'PYEOF'
import os
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model


BASE_MODEL = os.environ.get("BASE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
BASE_DIR = Path(os.environ.get("BASE_DIR", "/workspace"))
DATA_PATH = Path(os.environ.get("DATA_PATH", str(BASE_DIR / "data" / "train_chatml.jsonl")))
OUT_DIR = Path(os.environ.get("OUT_DIR", str(BASE_DIR / "models" / "lora_simple_rocm_out")))

MAX_TRAIN_ROWS = int(os.environ.get("MAX_TRAIN_ROWS", "2000"))
EPOCHS = int(os.environ.get("EPOCHS", "1"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
LR = float(os.environ.get("LR", "2e-4"))
SEED = int(os.environ.get("SEED", "42"))
USE_FP16 = os.environ.get("USE_FP16", "1") == "1"
USE_BF16 = os.environ.get("USE_BF16", "0") == "1"

LORA_R = int(os.environ.get("LORA_R", "8"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "16"))

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing training file: {DATA_PATH}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)

print(f"Loading tokenizer: {BASE_MODEL}")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(f"Loading dataset: {DATA_PATH}")
ds = load_dataset("json", data_files=str(DATA_PATH), split="train")
if MAX_TRAIN_ROWS > 0 and len(ds) > MAX_TRAIN_ROWS:
    ds = ds.select(range(MAX_TRAIN_ROWS))

def to_text(batch):
    texts = []
    for msgs in batch["messages"]:
        texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
    return {"text": texts}

ds = ds.shuffle(seed=SEED)
ds = ds.map(to_text, batched=True, remove_columns=ds.column_names)

def tok_fn(batch):
    return tok(batch["text"], truncation=True, max_length=SEQ_LEN, padding=False)

ds = ds.map(tok_fn, batched=True, remove_columns=["text"])

print(f"Loading model: {BASE_MODEL}")
dtype = torch.bfloat16 if USE_BF16 else (torch.float16 if USE_FP16 else torch.float32)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=dtype,
    token=HF_TOKEN,
)
model.config.use_cache = False

print("Applying LoRA adapters")
peft_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_cfg)

collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, pad_to_multiple_of=8)

args = TrainingArguments(
    output_dir=str(OUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    logging_steps=10,
    save_steps=250,
    save_total_limit=2,
    save_strategy="steps",
    report_to=[],
    fp16=USE_FP16,
    bf16=USE_BF16,
    remove_unused_columns=False,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=collator,
    processing_class=tok,
)

print("Starting training")
trainer.train()

print(f"Saving LoRA adapter to: {OUT_DIR}")
model.save_pretrained(str(OUT_DIR))
tok.save_pretrained(str(OUT_DIR))
print("Training complete")
PYEOF

echo ''
echo 'Training Complete!'
echo "Model saved to: ${OUT_DIR_RUN}"
CMD
)

if [ "$MODE" = "background" ]; then
  banner "Phase 2: Start Background Training"
  out "  ${WHITE}${BOLD}Container${RESET}  $CONTAINER_NAME"
  out "  ${WHITE}${BOLD}Log File${RESET}   $TRAIN_LOG"
  out "  ${WHITE}${BOLD}Rows${RESET}       $MAX_TRAIN_ROWS_RUN"
  out "  ${WHITE}${BOLD}Output${RESET}     $OUT_DIR_RUN"
  echo ""

  rm -f "$TRAIN_LOG"

  nohup docker run --rm \
    --name "$CONTAINER_NAME" \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --ipc=host \
    --shm-size 8G \
    -v "$(pwd)":/workspace \
    -v "$HF_CACHE_DIR":/root/.cache/huggingface \
    -w /workspace \
    -e HF_TOKEN="$HF_TOKEN" \
    -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
    "$IMAGE_NAME" bash -lc "$run_cmd" > "$TRAIN_LOG" 2>&1 &

  status_ok "Training started in background"
  out "  Check status: ./scripts/phase2_training.sh --status"
  out "  Follow logs : tail -f $TRAIN_LOG"
  exit 0
fi

banner "Phase 2: Start Training (Interactive)"
out "  ${WHITE}${BOLD}Rows${RESET}    $MAX_TRAIN_ROWS_RUN"
out "  ${WHITE}${BOLD}Output${RESET}  $OUT_DIR_RUN"

docker run -it --rm \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --ipc=host \
  --shm-size 8G \
  -v "$(pwd)":/workspace \
  -v "$HF_CACHE_DIR":/root/.cache/huggingface \
  -w /workspace \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
  "$IMAGE_NAME" bash -lc "$run_cmd"

echo ""
status_ok "Training complete"
out "  Model saved to host path mapped from: $OUT_DIR_RUN"
