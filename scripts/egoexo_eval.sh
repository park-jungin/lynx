#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

MODEL_SIZE=${MODEL_SIZE:-0.5b}
MODEL_PATH=${MODEL_PATH:-llava-hf/llava-onevision-qwen2-${MODEL_SIZE}-ov-hf}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-True}

VISION_ADAPTER_PATH=${VISION_ADAPTER_PATH:-}
LLM_ADAPTER_PATH=${LLM_ADAPTER_PATH:-}

DATA_ROOT=${DATA_ROOT:-./data/video_instruction_tuning/egoexo}
TAKE_TO_VIDEO_MAP=${TAKE_TO_VIDEO_MAP:-${DATA_ROOT}/from_take_id_to_video.json}
ANNOTATION_FILE=${ANNOTATION_FILE:-${DATA_ROOT}/proficiency_demonstrator_train_instruct.json}

PRED_SAVE=${PRED_SAVE:-./data/video_instruction_tuning/prediction/lynx_egoexo_predictions.json}

NUM_FRAMES=${NUM_FRAMES:-20}
FRAME_CHUNK_SIZE=${FRAME_CHUNK_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
MAX_SAMPLES=${MAX_SAMPLES:-}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$(dirname "${PRED_SAVE}")"

EXTRA_ARGS=()
LOCAL_FILES_ARGS=()
if [[ "${LOCAL_FILES_ONLY}" == "True" || "${LOCAL_FILES_ONLY}" == "true" || "${LOCAL_FILES_ONLY}" == "1" ]]; then
  LOCAL_FILES_ARGS+=(--local_files_only)
else
  LOCAL_FILES_ARGS+=(--no_local_files_only)
fi
if [ -n "${VISION_ADAPTER_PATH}" ]; then
  EXTRA_ARGS+=(--vision_adapter_path "${VISION_ADAPTER_PATH}")
fi
if [ -n "${LLM_ADAPTER_PATH}" ]; then
  EXTRA_ARGS+=(--llm_adapter_path "${LLM_ADAPTER_PATH}")
fi
if [ -n "${MAX_SAMPLES}" ]; then
  EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python scripts/egoexo_eval.py \
  --model_name_or_path "${MODEL_PATH}" \
  "${LOCAL_FILES_ARGS[@]}" \
  --annotation_file "${ANNOTATION_FILE}" \
  --video_root "${DATA_ROOT}" \
  --video_mapping "${TAKE_TO_VIDEO_MAP}" \
  --pred_save "${PRED_SAVE}" \
  --num_frames "${NUM_FRAMES}" \
  --frame_chunk_size "${FRAME_CHUNK_SIZE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  "${EXTRA_ARGS[@]}"

python tools/egoexo/calculate_egoexo_accuracy.py --prediction-path "${PRED_SAVE}"
