#!/usr/bin/env sh
# set -euo pipefail

# Usage:
#   ./run_qwen3vl.sh [input_jsonl] [output_jsonl] [image_root] [extra args...]
# Example:
#   ./run_qwen3vl.sh data.jsonl scored.jsonl . --batch-size 2

INPUT_JSONL="${1:-/kaggle/input/datasets/trnlqung/vitext-vqa/ViTextVQA_train.json}"
OUTPUT_JSONL="${2:-scored.json}"
IMAGE_ROOT="${3:-/kaggle/input/datasets/trnlqung/vitext-vqa/ViTextVQA_images/st_images}"
MODEL="Qwen/Qwen3-VL-4B-Instruct"

if [ "$#" -ge 3 ]; then
  shift 3
else
  shift "$#"
fi

python main/qwen3vl_vqa_difficulty.py --input "${INPUT_JSONL}" --output "${OUTPUT_JSONL}" --image-root "${IMAGE_ROOT}" --model "${MODEL}" --batch-size 1 --max-new-tokens 192 --short-side 768 --long-side 1280 --max-pixels 983040 --double-quant "$@"