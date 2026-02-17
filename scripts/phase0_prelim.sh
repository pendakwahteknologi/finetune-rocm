#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/cli_theme.sh"

BASE_IMAGE="${BASE_IMAGE:-rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1}"
IMAGE_NAME="${IMAGE_NAME:-finetune-rocm:ready}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
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

banner "Phase 0: Prelim Setup"
out "  ${WHITE}${BOLD}Steps${RESET}  Verify environment, build image, cache base model"
out "  ${WHITE}${BOLD}Image${RESET}  $IMAGE_NAME"
out "  ${WHITE}${BOLD}Model${RESET}  $BASE_MODEL"
out "  ${WHITE}${BOLD}TZ${RESET}     $CONTAINER_TZ"
echo ""

out "  ${YELLOW}[1/4] Verifying environment${RESET}"
VERIFY_FROM_PRELIM=1 "$ROOT_DIR/scripts/verify_environment.sh"
echo ""

mkdir -p "$HF_CACHE_DIR"

if [ "$FORCE_REBUILD" = "1" ]; then
  status_warn "FORCE_REBUILD=1 -> removing image: $IMAGE_NAME"
  docker image rm "$IMAGE_NAME" >/dev/null 2>&1 || true
fi

if docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  status_ok "[2/4] Image exists: $IMAGE_NAME (skip pull/build)"
else
  out "  ${YELLOW}[2/4] Pulling base image${RESET}  $BASE_IMAGE"
  docker pull "$BASE_IMAGE"

  out "  ${YELLOW}[3/4] Building training image${RESET}  $IMAGE_NAME"
  docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$IMAGE_NAME" -f - . <<'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN pip install --upgrade pip && \
    pip install transformers datasets peft accelerate sentencepiece protobuf
DOCKERFILE
fi

out "  ${YELLOW}[4/4] Caching base model${RESET}  $BASE_MODEL"
docker run --rm \
  "${TZ_DOCKER_ARGS[@]}" \
  -v "$HF_CACHE_DIR":/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
  "$IMAGE_NAME" python -c "
from huggingface_hub import snapshot_download
snapshot_download('$BASE_MODEL', token='$HF_TOKEN')
print('Model cache ready')
"

echo ""
status_ok "Prelim complete"
out "  Next:"
out "    ./scripts/start_finetuning_process.sh prepare"
out "    ./scripts/start_finetuning_process.sh train"
out "    ./scripts/start_finetuning_process.sh compare"
out "    ./scripts/start_finetuning_process.sh gguf"
