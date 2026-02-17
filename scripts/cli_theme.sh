#!/bin/bash
# Shared CLI theme for finetune scripts.

BOLD="\033[1m"
DIM="\033[2m"
RESET="\033[0m"
RED="\033[1;31m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
BLUE="\033[1;34m"
MAGENTA="\033[1;35m"
CYAN="\033[1;36m"
WHITE="\033[1;37m"
GRAY="\033[0;37m"

supports_color() {
  [ -t 1 ] && [ "${NO_COLOR:-}" = "" ]
}

if ! supports_color; then
  BOLD=""
  DIM=""
  RESET=""
  RED=""
  GREEN=""
  YELLOW=""
  BLUE=""
  MAGENTA=""
  CYAN=""
  WHITE=""
  GRAY=""
fi

term_width() {
  local w
  w="$(tput cols 2>/dev/null || echo 100)"
  if [ "$w" -lt 72 ]; then
    w=72
  elif [ "$w" -gt 120 ]; then
    w=120
  fi
  echo "$w"
}

CLI_WIDTH="$(term_width)"
KV_KEY_WIDTH=18

out() {
  echo -e "$1"
}

draw_line() {
  local char="${1:-=}"
  printf "%${CLI_WIDTH}s" "" | tr " " "$char"
  echo ""
}

banner() {
  local title="$1"
  local subtitle="${2:-}"
  out "${CYAN}$(draw_line)${RESET}"
  printf "%b%*s%b\n" "${WHITE}${BOLD}" $(( (CLI_WIDTH + ${#title}) / 2 )) "$title" "${RESET}"
  if [ -n "$subtitle" ]; then
    printf "%b%*s%b\n" "${DIM}" $(( (CLI_WIDTH + ${#subtitle}) / 2 )) "$subtitle" "${RESET}"
  fi
  out "${CYAN}$(draw_line)${RESET}"
}

section() {
  local title="$1"
  out "${CYAN}$(draw_line '-')${RESET}"
  out "${WHITE}${BOLD}${title}${RESET}"
  out "${CYAN}$(draw_line '-')${RESET}"
}

kv() {
  local key="$1"
  shift
  local value="$*"
  printf "  %b%-${KV_KEY_WIDTH}s%b %s\n" "${DIM}" "$key" "${RESET}" "$value"
}

step() {
  local current="$1"
  local total="$2"
  shift 2
  out "  ${YELLOW}[${current}/${total}]${RESET} $*"
}

tip() {
  out "  ${BLUE}TIP${RESET}  $1"
}

status_ok() {
  out "${GREEN}[OK]${RESET}   $1"
}

status_warn() {
  out "${YELLOW}[WARN]${RESET} $1"
}

status_err() {
  out "${RED}[ERROR]${RESET} $1"
}
