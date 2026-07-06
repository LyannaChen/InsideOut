#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="${MODEL_NAME:-chatgpt}"
OUTPUT_PATH="${OUTPUT_PATH:-${ROOT_DIR}/outputs/basic/${MODEL_NAME}/}"
OUTPUT_FILE_NAME="${OUTPUT_FILE_NAME:-generated_scripts_basic_${MODEL_NAME}.csv}"
SELECTION_FILE="${SELECTION_FILE:-}"

cmd=(
  python "${ROOT_DIR}/generate_contexted_docs.py"
  -m "${MODEL_NAME}"
  -o "${OUTPUT_PATH}"
  -f "${OUTPUT_FILE_NAME}"
  -p basic
  --batch_size "${BATCH_SIZE:-4}"
)

if [[ -n "${SELECTION_FILE}" ]]; then
  cmd+=(--selection_file "${SELECTION_FILE}")
fi

"${cmd[@]}"
