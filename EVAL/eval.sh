#!/usr/bin/env bash
set -euo pipefail
# random sample 1000
# hat 1500
# grpo hte 3000
CKPT_ROOT="/home/yangjiacheng/data/jiarui/HARI-Qwen3/HARI"
LOG_ROOT="/home/yangjiacheng/data/jiarui/HARI-Qwen3/EVAL-all"
RESULT_SAVE_PATH="/home/yangjiacheng/data/jiarui/EVAL/Qwen-MME"


# CKPT_ROOT="/home/yangjiacheng/data/jiarui/GRPO/TRAIN/GRPO-more"
# LOG_ROOT="/home/yangjiacheng/data/jiarui/GRPO/EVAL-all"
# RESULT_SAVE_PATH="/home/yangjiacheng/data/jiarui/EVAL/GRPO-MME"

LOG_DIR="${LOG_ROOT}/MME-logs"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULT_SAVE_PATH"
TS="$(date +%Y%m%d_%H%M%S)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MME_PY="$SCRIPT_DIR/MME.py"

found_any=0
for ckpt in "$CKPT_ROOT"/checkpoint-*; do
  [ -d "$ckpt" ] || continue
  step="${ckpt##*-}"
  if [[ ! "$step" =~ ^[0-9]+$ ]]; then
    continue
  fi
  
  # if (( step%3!=0 )); then
  #   continue
  # fi

  if (( step != 21135 )); then
    continue
  fi

  found_any=1
  log_file="$LOG_DIR/checkpoint-${step}_${TS}_MME.log"

  echo "==> checkpoint-${step} | log: ${log_file}"

  # for i in 1; do
  CUDA_VISIBLE_DEVICES="0" TRAINED_MODEL_ID="$ckpt" RESULT_SAVE_PATH="$RESULT_SAVE_PATH" python "$MME_PY" >"$log_file" 2>&1
  # done

done

if (( found_any == 0 )); then
  echo "No checkpoint-* directories were found under $CKPT_ROOT" >&2
  exit 1
fi