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

out() {
  echo -e "$1"
}

draw_line() {
  local char="${1:-=}"
  printf '%0.s'"$char" {1..80}
  echo ""
}

banner() {
  local title="$1"
  out "${CYAN}$(draw_line)${RESET}"
  out "${WHITE}${BOLD}${title}${RESET}"
  out "${CYAN}$(draw_line)${RESET}"
}

section() {
  local title="$1"
  out "${CYAN}$(draw_line '-')${RESET}"
  out "${WHITE}${BOLD}${title}${RESET}"
  out "${CYAN}$(draw_line '-')${RESET}"
}

status_ok() {
  out "${GREEN}OK${RESET}  $1"
}

status_warn() {
  out "${YELLOW}WARN${RESET}  $1"
}

status_err() {
  out "${RED}ERROR${RESET}  $1"
}
