#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE="${PIPELINE:-mfa-plan}"
GEN_MODEL="${GEN_MODEL:-chatgpt}"
EVAL_MODEL="${EVAL_MODEL:-o4-mini}"
FILE_PIPELINE="${PIPELINE//-/_}"
INPUT_FILE="${INPUT_FILE:-${ROOT_DIR}/outputs/${PIPELINE}/${GEN_MODEL}/generated_scripts_${FILE_PIPELINE}_${GEN_MODEL}.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-${ROOT_DIR}/evaluation_results/${PIPELINE}/${GEN_MODEL}/}"
OUTPUT_FILE_NAME="${OUTPUT_FILE_NAME:-eval_results_generated_scripts_${FILE_PIPELINE}_${GEN_MODEL}.csv}"

python "${ROOT_DIR}/evaluation/evaluate_outputs.py" \
  -m "${EVAL_MODEL}" \
  -i "${INPUT_FILE}" \
  -of "${OUTPUT_FILE_NAME}" \
  -o "${OUTPUT_PATH}" \
  -p "${PIPELINE}"
