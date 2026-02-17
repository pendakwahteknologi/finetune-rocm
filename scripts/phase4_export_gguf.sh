#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/cli_theme.sh"

BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
LORA_ADAPTER="${LORA_ADAPTER:-models/lora_simple_rocm_out}"
MERGED_DIR="${MERGED_DIR:-models/merged_llama31_8b_lora}"
GGUF_OUT_DIR="${GGUF_OUT_DIR:-models/gguf}"
GGUF_OUTTYPE="${GGUF_OUTTYPE:-f16}"
QUANTIZE_TYPE="${QUANTIZE_TYPE:-Q8_0}"
SKIP_QUANTIZE="${SKIP_QUANTIZE:-0}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT_DIR/llama.cpp}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
IMAGE_NAME="${IMAGE_NAME:-finetune-rocm:ready}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$ROOT_DIR/.cache/huggingface}"
CONTAINER_TZ="${CONTAINER_TZ:-Asia/Kuala_Lumpur}"
MERGE_DEVICE_MAP="${MERGE_DEVICE_MAP:-auto}"
CLEAN_MERGED_DIR="${CLEAN_MERGED_DIR:-1}"

TZ_DOCKER_ARGS=(-e "TZ=$CONTAINER_TZ")
if [ -f "/usr/share/zoneinfo/$CONTAINER_TZ" ]; then
  TZ_DOCKER_ARGS+=(-v "/usr/share/zoneinfo/$CONTAINER_TZ:/etc/localtime:ro")
elif [ -f /etc/localtime ]; then
  TZ_DOCKER_ARGS+=(-v /etc/localtime:/etc/localtime:ro)
fi
if [ -f /etc/timezone ]; then
  TZ_DOCKER_ARGS+=(-v /etc/timezone:/etc/timezone:ro)
fi

to_host_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    echo "$p"
  else
    echo "$ROOT_DIR/$p"
  fi
}

to_container_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    if [[ "$p" == "$ROOT_DIR" ]]; then
      echo "/workspace"
      return
    fi
    if [[ "$p" == "$ROOT_DIR/"* ]]; then
      echo "/workspace/${p#"$ROOT_DIR"/}"
      return
    fi
    status_err "Path must be inside repository for Docker phase 4: $p"
    out "  Use repo-relative paths or absolute paths under: $ROOT_DIR"
    exit 1
  fi
  echo "/workspace/$p"
}

LORA_ADAPTER_HOST="$(to_host_path "$LORA_ADAPTER")"
MERGED_DIR_HOST="$(to_host_path "$MERGED_DIR")"
GGUF_OUT_DIR_HOST="$(to_host_path "$GGUF_OUT_DIR")"
LORA_ADAPTER_CONTAINER="$(to_container_path "$LORA_ADAPTER")"
MERGED_DIR_CONTAINER="$(to_container_path "$MERGED_DIR")"
GGUF_OUT_DIR_CONTAINER="$(to_container_path "$GGUF_OUT_DIR")"

if ! command -v docker >/dev/null 2>&1; then
  status_err "Docker is required but not found in PATH."
  out "  Install Docker, then re-run phase 4."
  exit 1
fi

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  status_err "Pre-built image not found: $IMAGE_NAME"
  out "  Run ./scripts/start_finetuning_process.sh prelim first."
  exit 1
fi

if [ -z "$HF_TOKEN" ]; then
  status_err "HF token is required for GGUF export (base model merge step)."
  out "  Set it first:"
  out "    export HF_TOKEN=\"<your_hf_token>\""
  out "    export HUGGINGFACE_HUB_TOKEN=\"\$HF_TOKEN\""
  exit 1
fi

CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
QUANTIZE_BIN_DEFAULT="$LLAMA_CPP_DIR/build/bin/llama-quantize"
QUANTIZE_BIN_ALT="$LLAMA_CPP_DIR/llama-quantize"

if [ ! -f "$CONVERT_SCRIPT" ]; then
  status_err "Missing converter: $CONVERT_SCRIPT"
  out "  Set LLAMA_CPP_DIR to your llama.cpp checkout."
  exit 1
fi

if [ ! -d "$LORA_ADAPTER_HOST" ]; then
  status_err "LoRA adapter not found: $LORA_ADAPTER_HOST"
  out "  Run training first or set LORA_ADAPTER to an existing adapter path."
  exit 1
fi

mkdir -p "$HF_CACHE_DIR"
for dir in "$MERGED_DIR_HOST" "$GGUF_OUT_DIR_HOST"; do
  if ! mkdir -p "$dir" 2>/dev/null; then
    status_err "Cannot create directory: $dir (permission denied)"
    out "  This usually means earlier Docker phases created root-owned files."
    out "  Fix ownership from repo root, then re-run phase 4:"
    out "    sudo chown -R \"\$USER\":\"\$USER\" ."
    exit 1
  fi
done

banner "Phase 4: Export Merged GGUF"
out "  ${WHITE}${BOLD}Image${RESET}   $IMAGE_NAME"
out "  ${WHITE}${BOLD}TZ${RESET}      $CONTAINER_TZ"
out "  ${WHITE}${BOLD}Merge Map${RESET}  $MERGE_DEVICE_MAP"
out "  ${WHITE}${BOLD}Clean Merge${RESET}  $CLEAN_MERGED_DIR"
echo ""

if [ "$CLEAN_MERGED_DIR" = "1" ]; then
  out "  ${YELLOW}Cleaning previous merged output${RESET}  $MERGED_DIR_HOST"
  rm -rf "$MERGED_DIR_HOST"
  mkdir -p "$MERGED_DIR_HOST"
fi
echo ""

out "  ${YELLOW}[1/3] Merging LoRA adapter into base model${RESET}"
docker run --rm \
  -i \
  "${TZ_DOCKER_ARGS[@]}" \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)":/workspace \
  -v "$HF_CACHE_DIR":/hf_cache \
  -w /workspace \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
  -e HF_HOME=/hf_cache \
  -e TRANSFORMERS_CACHE=/hf_cache \
  -e HUGGINGFACE_HUB_CACHE=/hf_cache \
  -e BASE_MODEL="$BASE_MODEL" \
  -e LORA_ADAPTER="$LORA_ADAPTER_CONTAINER" \
  -e MERGED_DIR="$MERGED_DIR_CONTAINER" \
  -e MERGE_DTYPE="${MERGE_DTYPE:-float16}" \
  -e MERGE_DEVICE_MAP="$MERGE_DEVICE_MAP" \
  "$IMAGE_NAME" python - <<'PYEOF'
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(name: str):
    v = (name or "float16").lower()
    if v == "bfloat16":
        return torch.bfloat16
    if v == "float32":
        return torch.float32
    return torch.float16


base_model_name = os.environ.get("BASE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
lora_adapter = os.environ.get("LORA_ADAPTER", "models/lora_simple_rocm_out")
merged_dir = Path(os.environ.get("MERGED_DIR", "models/merged_llama31_8b_lora"))
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
dtype = resolve_dtype(os.environ.get("MERGE_DTYPE", "float16"))
device_map = os.environ.get("MERGE_DEVICE_MAP", "auto")

merged_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading base model: {base_model_name}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    dtype=dtype,
    device_map=device_map,
    token=hf_token,
)

print(f"Loading LoRA adapter: {lora_adapter}")
if not Path(lora_adapter).exists():
    raise FileNotFoundError(f"LoRA adapter path not found: {lora_adapter}")
model = PeftModel.from_pretrained(base_model, lora_adapter)

print("Merging adapter into base model...")
merged = model.merge_and_unload()

# Keep config metadata aligned with the original base architecture so
# convert_hf_to_gguf can detect the model family reliably.
base_model_type = getattr(base_model.config, "model_type", None)
base_architectures = getattr(base_model.config, "architectures", None)
if base_model_type:
    merged.config.model_type = base_model_type
if base_architectures:
    merged.config.architectures = list(base_architectures)
elif not getattr(merged.config, "architectures", None):
    merged.config.architectures = ["LlamaForCausalLM"]

print(f"Saving merged model to: {merged_dir}")
merged.save_pretrained(str(merged_dir), safe_serialization=True)

tokenizer_src = lora_adapter if Path(lora_adapter).exists() else base_model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, token=hf_token)
tokenizer.save_pretrained(str(merged_dir))

print(f"Saved config model_type: {getattr(merged.config, 'model_type', 'unknown')}")
print(f"Saved config architectures: {getattr(merged.config, 'architectures', [])}")
print("Merge complete.")
PYEOF

F16_GGUF_HOST="$GGUF_OUT_DIR_HOST/model-${GGUF_OUTTYPE}.gguf"
F16_GGUF_CONTAINER="$GGUF_OUT_DIR_CONTAINER/model-${GGUF_OUTTYPE}.gguf"
out "  ${YELLOW}[2/3] Converting merged model to GGUF${RESET}  $F16_GGUF_HOST"
if [ ! -f "$MERGED_DIR_HOST/config.json" ]; then
  status_err "Missing merged config: $MERGED_DIR_HOST/config.json"
  out "  Merge step did not produce a valid HF model directory."
  if [ -d "$MERGED_DIR_HOST" ]; then
    out "  Current files in merged dir:"
    ls -1 "$MERGED_DIR_HOST" | sed 's/^/    - /'
  fi
  exit 1
fi
docker run --rm \
  "${TZ_DOCKER_ARGS[@]}" \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)":/workspace \
  -v "$LLAMA_CPP_DIR":/llama.cpp:ro \
  -w /workspace \
  "$IMAGE_NAME" python /llama.cpp/convert_hf_to_gguf.py "$MERGED_DIR_CONTAINER" --outfile "$F16_GGUF_CONTAINER" --outtype "$GGUF_OUTTYPE"

if [ "$SKIP_QUANTIZE" = "1" ]; then
  status_warn "SKIP_QUANTIZE=1, skipping quantization"
  status_ok "Done: $F16_GGUF_HOST"
  exit 0
fi

QUANTIZE_BIN="$QUANTIZE_BIN_DEFAULT"
if [ ! -x "$QUANTIZE_BIN" ] && [ -x "$QUANTIZE_BIN_ALT" ]; then
  QUANTIZE_BIN="$QUANTIZE_BIN_ALT"
fi

if [ ! -x "$QUANTIZE_BIN" ]; then
  status_warn "Quantizer not found in llama.cpp build."
  out "  Built GGUF (unquantized): $F16_GGUF_HOST"
  out "  Build llama.cpp and re-run for quantization."
  exit 0
fi

QUANTIZED_GGUF_HOST="$GGUF_OUT_DIR_HOST/model-${QUANTIZE_TYPE}.gguf"
out "  ${YELLOW}[3/3] Quantizing GGUF${RESET}  $QUANTIZE_TYPE"
"$QUANTIZE_BIN" "$F16_GGUF_HOST" "$QUANTIZED_GGUF_HOST" "$QUANTIZE_TYPE"

echo ""
status_ok "GGUF export complete"
out "  - $F16_GGUF_HOST"
out "  - $QUANTIZED_GGUF_HOST"
