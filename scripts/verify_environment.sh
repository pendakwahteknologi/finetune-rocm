#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/cli_theme.sh"

HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT_DIR/llama.cpp}"
MIN_FREE_GB="${MIN_FREE_GB:-60}"
VERIFY_FROM_PRELIM="${VERIFY_FROM_PRELIM:-0}"

FAILS=0
WARNS=0

pass() {
  status_ok "$1"
}

warn() {
  status_warn "$1"
  WARNS=$((WARNS + 1))
}

fail() {
  status_err "$1"
  FAILS=$((FAILS + 1))
}

banner "Prelim: Environment Verification"

if [ -f /etc/os-release ]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  if [ "${ID:-}" = "ubuntu" ] && [[ "${VERSION_ID:-}" == 24.04* ]]; then
    pass "OS check: Ubuntu ${VERSION_ID:-unknown}"
  elif [ "${ID:-}" = "ubuntu" ]; then
    warn "OS check: Ubuntu ${VERSION_ID:-unknown} (tested target is Ubuntu 24.04.x)"
  else
    warn "OS check: ${PRETTY_NAME:-unknown distro} (tested target is Ubuntu 24.04.x)"
  fi
else
  warn "OS check: /etc/os-release not found"
fi

if command -v docker >/dev/null 2>&1; then
  pass "Docker binary found"
else
  fail "Docker is not installed or not in PATH"
fi

if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    pass "Docker daemon is reachable"
  else
    fail "Docker daemon is not reachable for current user"
  fi
fi

if [ -e /dev/kfd ]; then
  pass "ROCm device present: /dev/kfd"
else
  fail "ROCm device missing: /dev/kfd"
fi

if [ -d /dev/dri ]; then
  pass "Graphics device directory present: /dev/dri"
else
  fail "Graphics device directory missing: /dev/dri"
fi

for grp in video render; do
  if id -nG 2>/dev/null | tr ' ' '\n' | grep -qx "$grp"; then
    pass "User is in group: $grp"
  else
    warn "User is not in group: $grp"
  fi
done

if [ -n "$HF_TOKEN" ]; then
  pass "Hugging Face token is set in environment"
  if [[ "$HF_TOKEN" != hf_* ]]; then
    warn "HF token format is unusual (expected prefix: hf_)"
  fi
else
  fail "HF token missing. Export HF_TOKEN (and HUGGINGFACE_HUB_TOKEN)"
fi

if command -v curl >/dev/null 2>&1; then
  if curl -fsS --max-time 10 https://huggingface.co >/dev/null 2>&1; then
    pass "Network check: huggingface.co reachable"
  else
    warn "Network check: unable to reach huggingface.co right now"
  fi
else
  warn "curl not found; skipped network reachability check"
fi

FREE_GB="$(df -Pk "$ROOT_DIR" | awk 'NR==2 {print int($4/1024/1024)}')"
if [ "${FREE_GB:-0}" -ge "$MIN_FREE_GB" ]; then
  pass "Disk space: ${FREE_GB}GB free (min ${MIN_FREE_GB}GB)"
else
  warn "Disk space: ${FREE_GB}GB free (recommended >= ${MIN_FREE_GB}GB)"
fi

CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
LLAMA_CLI="$LLAMA_CPP_DIR/build/bin/llama-cli"
LLAMA_MAIN="$LLAMA_CPP_DIR/main"
if [ -f "$CONVERT_SCRIPT" ]; then
  pass "llama.cpp converter found for Phase 4"
else
  warn "llama.cpp converter missing: $CONVERT_SCRIPT"
fi
if [ -x "$LLAMA_CLI" ] || [ -x "$LLAMA_MAIN" ]; then
  pass "llama.cpp chat binary found for Phase 5"
else
  warn "llama.cpp chat binary missing (build llama.cpp before Phase 5)"
fi

echo ""
if [ "$FAILS" -gt 0 ]; then
  status_err "Environment verification failed ($FAILS critical issue(s), $WARNS warning(s))."
  out "  Fix the critical issues above, then re-run:"
  out "    ./scripts/verify_environment.sh"
  exit 1
fi

status_ok "Environment verification passed ($WARNS warning(s))."
if [ "$VERIFY_FROM_PRELIM" = "1" ]; then
  out "  Continuing Phase 0 setup..."
else
  out "  Next:"
  out "    ./scripts/start_finetuning_process.sh prelim"
fi
