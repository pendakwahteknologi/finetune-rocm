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
  tip "Install Docker, then re-run phase 1."
  exit 1
fi

mkdir -p "$HF_CACHE_DIR"

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  status_err "Pre-built image not found: $IMAGE_NAME"
  tip "Run ./scripts/start_finetuning_process.sh prelim first."
  exit 1
fi

banner "Phase 1: Prepare Training Data"
kv "Image" "$IMAGE_NAME"
kv "OpenOrca rows" "$OPENORCA_N"
kv "Eval rows" "$EVAL_N"
kv "Seed" "$SEED"
kv "Timezone" "$CONTAINER_TZ"
echo ""

step 1 1 "Running data pipeline in Docker"
docker run --rm \
  "${TZ_DOCKER_ARGS[@]}" \
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
  -e HF_TOKEN="$HF_TOKEN" \
  -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
  "$IMAGE_NAME" python /workspace/scripts/phase1_prepare_data.py

if [ ! -f "data/train_chatml.jsonl" ]; then
  status_err "Expected output missing: data/train_chatml.jsonl"
  tip "Re-run prepare and ensure you are in the repository root."
  exit 1
fi

echo ""
status_ok "Data prep complete"
section "Outputs"
out "  - data/dolly15k_plus_openorca${OPENORCA_N}.jsonl"
out "  - data/train_clean.jsonl"
out "  - data/eval_prompts.jsonl"
out "  - data/train_chatml.jsonl"
