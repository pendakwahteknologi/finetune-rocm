#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/cli_theme.sh"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT_DIR/llama.cpp}"
MODEL_GGUF="${MODEL_GGUF:-$ROOT_DIR/models/gguf/model-Q8_0.gguf}"
MODEL_F16="${MODEL_F16:-$ROOT_DIR/models/gguf/model-f16.gguf}"
CTX_SIZE="${CTX_SIZE:-8192}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
TEMP="${TEMP:-0.7}"
TOP_P="${TOP_P:-0.9}"
DEFAULT_THREADS="$(command -v nproc >/dev/null 2>&1 && nproc || getconf _NPROCESSORS_ONLN || echo 8)"
THREADS="${THREADS:-$DEFAULT_THREADS}"

LLAMA_CLI="$LLAMA_CPP_DIR/build/bin/llama-cli"
LLAMA_MAIN="$LLAMA_CPP_DIR/main"

if [ ! -f "$MODEL_GGUF" ] && [ -f "$MODEL_F16" ]; then
  MODEL_GGUF="$MODEL_F16"
fi

if [ ! -f "$MODEL_GGUF" ]; then
  status_err "No GGUF model found."
  out "  Expected one of:"
  out "    - $ROOT_DIR/models/gguf/model-Q8_0.gguf"
  out "    - $ROOT_DIR/models/gguf/model-f16.gguf"
  out "  Run Phase 4 first:"
  out "    ./scripts/start_finetuning_process.sh gguf"
  exit 1
fi

if [ -x "$LLAMA_CLI" ]; then
  BIN="$LLAMA_CLI"
elif [ -x "$LLAMA_MAIN" ]; then
  BIN="$LLAMA_MAIN"
else
  status_err "llama.cpp binary not found."
  out "  Checked:"
  out "    - $LLAMA_CLI"
  out "    - $LLAMA_MAIN"
  out "  Build llama.cpp and/or set LLAMA_CPP_DIR."
  exit 1
fi

banner "Phase 5: Chat with Fine-Tuned Model (llama.cpp)"
out "  ${WHITE}${BOLD}Binary${RESET}      $BIN"
out "  ${WHITE}${BOLD}Model${RESET}       $MODEL_GGUF"
out "  ${WHITE}${BOLD}Context${RESET}     $CTX_SIZE"
out "  ${WHITE}${BOLD}GPU Layers${RESET}  $N_GPU_LAYERS"
echo ""
out "  ${DIM}Type your prompt and press Enter. Ctrl+C to quit.${RESET}"
echo ""

exec "$BIN" \
  -m "$MODEL_GGUF" \
  -c "$CTX_SIZE" \
  --n-gpu-layers "$N_GPU_LAYERS" \
  --temp "$TEMP" \
  --top-p "$TOP_P" \
  -t "$THREADS" \
  -cnv "$@"
