#!/usr/bin/env sh
# set -euo pipefail

# Usage:
#   ./run_qwen3vl.sh [input_jsonl] [output_jsonl] [image_root] [extra args...]
# Example:
#   ./run_qwen3vl.sh data.jsonl scored.jsonl . --batch-size 2
#   ./run_qwen3vl.sh data_a.jsonl scored.jsonl . --input-2 data_b.jsonl --range-1 0:1000 --range-2 200:1200 --pick-1 300 --pick-2 300 --seed 123

INPUT_JSON="${1:-/kaggle/input/datasets/trnlqung/vitext-vqa/ViTextVQA_train.json}"
INPUT_JSON_2="${2:-/kaggle/input/datasets/trnlqung/openvivqa/openvivqa_train_v2.json}"
OUTPUT_JSON="${3:-scored.json}"
IMAGE_ROOT="${4:-/kaggle/input/datasets/trnlqung/vitext-vqa/ViTextVQA_images/st_images}"
IMAGE_ROOT_2="${5:-/kaggle/input/datasets/trnlqung/openvivqa/images/images}"
MODEL="Qwen/Qwen3-VL-4B-Instruct"
RANGE_1="0:3600"
RANGE_2="0:1800"
SEED="42"

if [ "$#" -ge 3 ]; then
  shift 3
else
  shift "$#"
fi

python main/qwen3vl_vqa_difficulty.py --input "${INPUT_JSON}" --input-2 "${INPUT_JSON_2}" --output "${OUTPUT_JSON}" --image-root "${IMAGE_ROOT}" --image-root-2 "${IMAGE_ROOT_2}" --range-1 "${RANGE_1}" --range-2 "${RANGE_2}" --model "${MODEL}" --batch-size 4 --max-new-tokens 192 --short-side 768 --long-side 1280 --max-pixels 983040 --double-quant --seed "${SEED}" "$@"
