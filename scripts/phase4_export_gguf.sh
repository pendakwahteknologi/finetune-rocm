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

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  status_err "Python interpreter not found (python/python3)."
  out "  Install Python 3, then re-run phase 4:"
  out "    sudo apt update && sudo apt install -y python3 python3-pip"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  status_err "Docker is required but not found in PATH."
  out "  Install Docker, then re-run phase 4."
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

for dir in "$MERGED_DIR" "$GGUF_OUT_DIR"; do
  if ! mkdir -p "$dir" 2>/dev/null; then
    status_err "Cannot create directory: $dir (permission denied)"
    out "  This usually means earlier Docker phases created root-owned files."
    out "  Fix ownership from repo root, then re-run phase 4:"
    out "    sudo chown -R \"\$USER\":\"\$USER\" ."
    exit 1
  fi
done

banner "Phase 4: Export Merged GGUF"
out "  ${WHITE}${BOLD}Python${RESET}  $PYTHON_BIN"
echo ""

out "  ${YELLOW}[1/3] Merging LoRA adapter into base model${RESET}"
BASE_MODEL="$BASE_MODEL" \
LORA_ADAPTER="$LORA_ADAPTER" \
MERGED_DIR="$MERGED_DIR" \
"$PYTHON_BIN" - <<'PYEOF'
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

merged_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading base model: {base_model_name}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    dtype=dtype,
    device_map="auto",
    token=hf_token,
)

print(f"Loading LoRA adapter: {lora_adapter}")
model = PeftModel.from_pretrained(base_model, lora_adapter)

print("Merging adapter into base model...")
merged = model.merge_and_unload()

print(f"Saving merged model to: {merged_dir}")
merged.save_pretrained(str(merged_dir), safe_serialization=True)

tokenizer_src = lora_adapter if Path(lora_adapter).exists() else base_model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, token=hf_token)
tokenizer.save_pretrained(str(merged_dir))

print("Merge complete.")
PYEOF

F16_GGUF="$GGUF_OUT_DIR/model-${GGUF_OUTTYPE}.gguf"
out "  ${YELLOW}[2/3] Converting merged model to GGUF${RESET}  $F16_GGUF"
"$PYTHON_BIN" "$CONVERT_SCRIPT" "$MERGED_DIR" --outfile "$F16_GGUF" --outtype "$GGUF_OUTTYPE"

if [ "$SKIP_QUANTIZE" = "1" ]; then
  status_warn "SKIP_QUANTIZE=1, skipping quantization"
  status_ok "Done: $F16_GGUF"
  exit 0
fi

QUANTIZE_BIN="$QUANTIZE_BIN_DEFAULT"
if [ ! -x "$QUANTIZE_BIN" ] && [ -x "$QUANTIZE_BIN_ALT" ]; then
  QUANTIZE_BIN="$QUANTIZE_BIN_ALT"
fi

if [ ! -x "$QUANTIZE_BIN" ]; then
  status_warn "Quantizer not found in llama.cpp build."
  out "  Built GGUF (unquantized): $F16_GGUF"
  out "  Build llama.cpp and re-run for quantization."
  exit 0
fi

QUANTIZED_GGUF="$GGUF_OUT_DIR/model-${QUANTIZE_TYPE}.gguf"
out "  ${YELLOW}[3/3] Quantizing GGUF${RESET}  $QUANTIZE_TYPE"
"$QUANTIZE_BIN" "$F16_GGUF" "$QUANTIZED_GGUF" "$QUANTIZE_TYPE"

echo ""
status_ok "GGUF export complete"
out "  - $F16_GGUF"
out "  - $QUANTIZED_GGUF"
