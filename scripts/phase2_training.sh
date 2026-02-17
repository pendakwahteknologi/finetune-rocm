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
ATTN_IMPL_DEFAULT="${ATTN_IMPL:-eager}"
GRADIENT_CHECKPOINTING_DEFAULT="${GRADIENT_CHECKPOINTING:-1}"
PYTORCH_ALLOC_CONF_DEFAULT="${PYTORCH_ALLOC_CONF:-${PYTORCH_HIP_ALLOC_CONF:-max_split_size_mb:64,garbage_collection_threshold:0.7}}"
OPTIMIZER_DEFAULT="${OPTIMIZER:-adamw_hf}"
DATALOADER_PIN_MEMORY_DEFAULT="${DATALOADER_PIN_MEMORY:-0}"

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
export ATTN_IMPL=${ATTN_IMPL_DEFAULT}
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING_DEFAULT}
export OPTIMIZER=${OPTIMIZER_DEFAULT}
export DATALOADER_PIN_MEMORY=${DATALOADER_PIN_MEMORY_DEFAULT}
export BASE_DIR=/workspace
export DATA_DIR=/workspace/data
export OUT_DIR=${OUT_DIR_RUN}

echo 'Testing GPU...'
python -c 'import torch; print(f"GPU: {torch.cuda.get_device_name(0)}"); x=torch.randn(100,100,device="cuda"); print("GPU test OK")'

echo 'Starting training...'
python /workspace/scripts/phase2_train_simple_rocm.py

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
  out "  ${WHITE}${BOLD}Attn Impl${RESET}  $ATTN_IMPL_DEFAULT"
  out "  ${WHITE}${BOLD}Optim${RESET}      $OPTIMIZER_DEFAULT"
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
out "  ${WHITE}${BOLD}Attn${RESET}    $ATTN_IMPL_DEFAULT"
out "  ${WHITE}${BOLD}Optim${RESET}   $OPTIMIZER_DEFAULT"
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
