#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE="${PIPELINE:-mfa-plan}"
GEN_MODEL="${GEN_MODEL:-chatgpt}"
FILE_PIPELINE="${PIPELINE//-/_}"
INPUT_FILE="${INPUT_FILE:-${ROOT_DIR}/evaluation_results/${PIPELINE}/${GEN_MODEL}/eval_results_generated_scripts_${FILE_PIPELINE}_${GEN_MODEL}.csv}"
OUTPUT_FILE="${OUTPUT_FILE:-${ROOT_DIR}/evaluation_results/${PIPELINE}/${GEN_MODEL}/summary_yes_percentage.csv}"

python "${ROOT_DIR}/evaluation/evaluate_results.py" \
  -i "${INPUT_FILE}" \
  -o "${OUTPUT_FILE}"
