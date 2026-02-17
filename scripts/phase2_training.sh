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
SEQ_LEN_DEFAULT="${SEQ_LEN:-256}"
PYTORCH_ALLOC_CONF_DEFAULT="${PYTORCH_ALLOC_CONF:-${PYTORCH_HIP_ALLOC_CONF:-max_split_size_mb:64,garbage_collection_threshold:0.7}}"

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
SEQ_LEN_RUN="$SEQ_LEN_DEFAULT"
if [ "$FULL_RUN" = "1" ]; then
  MAX_TRAIN_ROWS_RUN="${MAX_TRAIN_ROWS_FULL:-20011}"
  OUT_DIR_RUN="${OUT_DIR_FULL:-/workspace/models/lora_full_20k_rocm_out}"
  if [ -z "${SEQ_LEN:-}" ]; then
    SEQ_LEN_RUN="192"
  fi
fi

run_cmd=$(cat <<CMD
set -euo pipefail

# ROCm stability settings
export GPU_MAX_HW_QUEUES=2
export HSA_ENABLE_SDMA=0
unset PYTORCH_HIP_ALLOC_CONF || true
export PYTORCH_ALLOC_CONF='${PYTORCH_ALLOC_CONF_DEFAULT}'
export TOKENIZERS_PARALLELISM=false

# Training config
export USE_FP16=${USE_FP16_DEFAULT}
export USE_BF16=${USE_BF16_DEFAULT}
export BATCH_SIZE=${BATCH_SIZE_DEFAULT}
export GRAD_ACCUM=${GRAD_ACCUM_DEFAULT}
export MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS_RUN}
export SEQ_LEN=${SEQ_LEN_RUN}
export DATALOADER_WORKERS=${DATALOADER_WORKERS_DEFAULT}
export BASE_DIR=/workspace
export DATA_DIR=/workspace/data
export OUT_DIR=${OUT_DIR_RUN}

echo 'Testing GPU...'
python -c 'import torch; print(f"GPU: {torch.cuda.get_device_name(0)}"); x=torch.randn(100,100,device="cuda"); print("GPU test OK")'

echo 'Starting training...'
python - <<'PYEOF'
import gc
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


BASE_MODEL = os.environ.get("BASE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_HUB_TOKEN", "")

BASE_DIR = Path(os.environ.get("BASE_DIR", str(Path.home() / "finetune-rocm")))
DATA_DIR = BASE_DIR / "data"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(BASE_DIR / "models" / "lora_simple_rocm_out")))
LOG_DIR = BASE_DIR / "results"
CHAT_JSONL = DATA_DIR / "train_chatml.jsonl"

MAX_TRAIN_ROWS = int(os.environ.get("MAX_TRAIN_ROWS", "2000"))
EPOCHS = int(os.environ.get("EPOCHS", "1"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
LR = float(os.environ.get("LR", "2e-4"))
USE_FP16 = os.environ.get("USE_FP16", "1") == "1"

LORA_R = int(os.environ.get("LORA_R", "8"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "16"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "10"))

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.1")
os.environ.setdefault("GPU_MAX_HW_QUEUES", "2")
os.environ.setdefault("HSA_ENABLE_SDMA", "0")


def ensure_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        return torch.device("cuda")
    print("[warn] GPU unavailable or disabled - falling back to CPU")
    return torch.device("cpu")


def fmt_bytes(b):
    return f"{b / 1024**3:.2f} GiB"


def print_gpu_info():
    if not torch.cuda.is_available():
        return
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    if hasattr(props, "gcnArchName"):
        print(f"Arch: {props.gcnArchName}")
    print(f"Total VRAM: {fmt_bytes(props.total_memory)}")


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=True,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(dtype):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype,
        device_map=None,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    model.config.use_cache = False
    return model


def collate_dynamic_pad(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for x in batch:
        ids = x["input_ids"]
        lab = x["labels"]
        pad_n = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_n)
        attention_mask.append([1] * len(ids) + [0] * pad_n)
        labels.append(lab + [-100] * pad_n)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ensure_dirs()
    start = datetime.now()
    print(f"Start: {start:%Y-%m-%d %H:%M:%S}")

    device = get_device()
    dtype = torch.float16 if USE_FP16 else torch.float32
    print_gpu_info()
    print(f"Base model: {BASE_MODEL}")
    print(f"Data file: {CHAT_JSONL}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Seq len: {SEQ_LEN}, batch: {BATCH_SIZE}, grad_accum: {GRAD_ACCUM}")

    tokenizer = load_tokenizer()

    ds = load_dataset("json", data_files=str(CHAT_JSONL), split="train")
    if MAX_TRAIN_ROWS > 0 and len(ds) > MAX_TRAIN_ROWS:
        ds = ds.select(range(MAX_TRAIN_ROWS))
    print(f"Dataset rows selected: {len(ds)}")

    texts = []
    for ex in ds:
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    tokenized = []
    lengths = []
    for text in texts:
        enc = tokenizer(text, truncation=True, max_length=SEQ_LEN, padding=False)
        ids = enc["input_ids"]
        lengths.append(len(ids))
        tokenized.append({"input_ids": ids, "labels": ids.copy()})

    print(f"Avg tokens: {statistics.mean(lengths):.0f}, max: {max(lengths)}")

    loader = DataLoader(
        tokenized,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        collate_fn=lambda b: collate_dynamic_pad(b, tokenizer.pad_token_id),
    )
    print(f"Batches per epoch: {len(loader)}")

    model = load_model(dtype)
    model.to(device)

    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )
    model = get_peft_model(model, lora_config)
    model.train()
    print("LoRA adapters applied")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        foreach=False,
    )

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * EPOCHS
    print(f"Total training steps: {total_steps}")

    start_time = time.time()
    global_step = 0
    loss_window = []

    for epoch in range(1, EPOCHS + 1):
        for batch in loader:
            global_step += 1

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss_scalar = float(loss.detach().cpu().item())
            loss_window.append(loss_scalar)
            if len(loss_window) > LOG_EVERY:
                loss_window.pop(0)

            (loss / GRAD_ACCUM).backward()

            if global_step % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                avg_loss = statistics.mean(loss_window)
                sps = (global_step * BATCH_SIZE) / max(elapsed, 1e-9)
                print(
                    f"step={global_step}/{total_steps} "
                    f"epoch={epoch}/{EPOCHS} "
                    f"loss={avg_loss:.4f} "
                    f"samples_per_sec={sps:.2f}"
                )

    model_cpu = model.to("cpu")
    cleanup()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_cpu.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.1f}s")
    print(f"Saved adapter: {OUT_DIR}")


if __name__ == "__main__":
    main()
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
  out "  ${WHITE}${BOLD}Seq Len${RESET}    $SEQ_LEN_RUN"
  out "  ${WHITE}${BOLD}Output${RESET}     $OUT_DIR_RUN"
  echo ""

  rm -f "$TRAIN_LOG"

  nohup docker run --rm \
    --name "$CONTAINER_NAME" \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ulimit memlock=-1 \
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
out "  ${WHITE}${BOLD}Seq Len${RESET}  $SEQ_LEN_RUN"
out "  ${WHITE}${BOLD}Output${RESET}  $OUT_DIR_RUN"

docker run -it --rm \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
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
