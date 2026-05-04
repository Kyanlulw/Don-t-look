#!/usr/bin/env sh
# set -euo pipefail

# Usage:
#   ./run_qwen3vl.sh [input_1] [input_2] [output_jsonl] [image_root_1] [image_root_2] [extra args...]
# Example:
#   ./run_qwen3vl.sh data_a.json data_b.json scored.json . . --batch-size 2
#   ./run_qwen3vl.sh data_a.jsonl scored.jsonl . --input-2 data_b.jsonl --range-1 0:1000 --range-2 200:1200 --pick-1 300 --pick-2 300 --seed 123

INPUT_JSON="${1:-/kaggle/input/datasets/trnlqung/openvivqa/openvivqa/non-text-vqa.json}"
# INPUT_JSON_2="${2:-/teamspace/studios/this_studio/.cache/kagglehub/datasets/trnlqung/openvivqa/versions/1/openvivqa_train_v2.json}"
OUTPUT_JSON="${3:-scored.json}"
IMAGE_ROOT="${4:-/kaggle/input/datasets/trnlqung/openvivqa/images/images}"
# IMAGE_ROOT_2="${5:-/teamspace/studios/this_studio/.cache/kagglehub/datasets/trnlqung/openvivqa/versions/1/images/images}"
MODEL="Qwen/Qwen3-VL-8B-Instruct"
RANGE_1="0:15000"
RANGE_2="3100:5000"
SEED="42"
DISABLE_RESUME="${DISABLE_RESUME:-1}"

RESUME_FLAG=""
if [ "${DISABLE_RESUME}" = "1" ]; then
  RESUME_FLAG="--disable-resume"
fi

if [ "$#" -ge 5 ]; then
  shift 5
else
  shift "$#"
fi

ACCELERATE_INIT_DEVICE=cpu python main/qwen3vl_vqa_difficulty.py --input "${INPUT_JSON}" --output "${OUTPUT_JSON}" --image-root "${IMAGE_ROOT}" --range-1 "${RANGE_1}"  --model "${MODEL}" --model-backend qwen --prompt-style qwen --batch-size 16 --max-new-tokens 192 --short-side 448 --long-side 747 --max-pixels 334656 --seed "${SEED}" ${RESUME_FLAG} "$@"
