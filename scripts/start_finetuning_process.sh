#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/cli_theme.sh"

usage() {
  cat <<USAGE
Usage:
  ./scripts/start_finetuning_process.sh prelim
  ./scripts/start_finetuning_process.sh prepare
  ./scripts/start_finetuning_process.sh train
  ./scripts/start_finetuning_process.sh compare
  ./scripts/start_finetuning_process.sh gguf
  ./scripts/start_finetuning_process.sh chat [-- <llama.cpp args>]

Phases:
  prelim  Phase 0: system/docker/base-model prep
  prepare  Phase 1: prepare data (single script, Docker)
  train    Phase 2: fine-tune model (Docker)
  compare  Phase 3: compare base vs fine-tuned (Docker)
  gguf     Phase 4: merge + export GGUF
  chat     Phase 5: chat with exported GGUF via llama.cpp
USAGE
}

PHASE="${1:-}"
shift || true

banner "Fine-Tuning Process Dispatcher"
out "  ${DIM}Selected phase: ${PHASE:-<none>}${RESET}"
echo ""

case "$PHASE" in
  prelim)
    ./scripts/phase0_prelim.sh "$@"
    ;;
  prepare)
    ./scripts/phase1_prepare_data.sh "$@"
    ;;
  train)
    ./scripts/phase2_training.sh "$@"
    ;;
  compare)
    ./scripts/phase3_comparison_report.sh "$@"
    ;;
  gguf)
    ./scripts/phase4_export_gguf.sh "$@"
    ;;
  chat)
    ./scripts/phase5_chat_llamacpp.sh "$@"
    ;;
  *)
    status_err "Unknown phase: ${PHASE:-<none>}"
    echo ""
    usage
    exit 1
    ;;
esac
