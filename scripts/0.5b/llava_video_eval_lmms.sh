#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

# Evaluate LynX fast/slow model on VideoMME, MVBench, MLVU using lmms_eval_start.py.
#
# Assumes datasets/videos are available under `HF_HOME` as expected by lmms_eval task YAMLs.
# Example:
#   export HF_HOME=/mnt/hdd1/datasets/video_instruction_tuning

MODEL_SIZE=${MODEL_SIZE:-0.5b}
MODEL_PATH=${MODEL_PATH:-llava-hf/llava-onevision-qwen2-${MODEL_SIZE}-ov-hf}

# Adapter paths:
# - VISION_ADAPTER_PATH: Stage-1/2 output dir (vision LoRA)
# - LLM_ADAPTER_PATH: Stage-3 SFT output dir (LLM LoRA)
VISION_ADAPTER_PATH=${VISION_ADAPTER_PATH:-""}
LLM_ADAPTER_PATH=${LLM_ADAPTER_PATH:-""}

SLOW_NUM_FRAMES=${SLOW_NUM_FRAMES:-20}
FAST_MULTIPLIER=${FAST_MULTIPLIER:-4}
FAST_NUM_FRAMES=${FAST_NUM_FRAMES:-$((SLOW_NUM_FRAMES * FAST_MULTIPLIER))}
FAST_TILE_SIZE=${FAST_TILE_SIZE:-4}
FRAME_CHUNK_SIZE=${FRAME_CHUNK_SIZE:-4}

TASKS=${TASKS:-"videomme,mvbench,mlvu"}
LIMIT=${LIMIT:-""}          # e.g. 50 for quick sanity, or empty for full
BATCH_SIZE=${BATCH_SIZE:-1} # keep 1 unless you know it fits

OUT_DIR=${OUT_DIR:-./checkpoints/eval/llava_video_${MODEL_SIZE}/$(date +%Y-%m-%d_%H%M%S)}
mkdir -p "${OUT_DIR}"

MODEL_ARGS="pretrained=${MODEL_PATH},local_files_only=True,cache_dir=/mnt/hdd1/,slow_num_frames=${SLOW_NUM_FRAMES},fast_num_frames=${FAST_NUM_FRAMES},fast_tile_size=${FAST_TILE_SIZE},frame_chunk_size=${FRAME_CHUNK_SIZE}"
if [ -n "${VISION_ADAPTER_PATH}" ]; then
  MODEL_ARGS="${MODEL_ARGS},vision_adapter_path=${VISION_ADAPTER_PATH},vision_adapter_name=vision"
fi
if [ -n "${LLM_ADAPTER_PATH}" ]; then
  MODEL_ARGS="${MODEL_ARGS},llm_adapter_path=${LLM_ADAPTER_PATH},llm_adapter_name=llm"
fi

LIMIT_ARGS=()
if [ -n "${LIMIT}" ]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

python lmms_eval_start.py \
  --model lynx_fastvideo_onevision \
  --model_args "${MODEL_ARGS}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  --log_samples \
  --output_path "${OUT_DIR}" \
  "${LIMIT_ARGS[@]}"
