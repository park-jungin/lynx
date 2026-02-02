MODEL_SIZE=${MODEL_SIZE:-7b}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-llava-hf/llava-onevision-qwen2-$MODEL_SIZE-ov-hf}
VISION_ADAPTER_PATH=${VISION_ADAPTER_PATH:-./checkpoints/LynX_$MODEL_SIZE/3d/sqa/train}
LLM_ADAPTER_PATH=${LLM_ADAPTER_PATH:-./checkpoints/LynX_$MODEL_SIZE/3d/sqa/sft}

OUTPUT_DIR=${OUTPUT_DIR:-./downstream_evaluation/3d}
PRED_SAVE=${PRED_SAVE:-$OUTPUT_DIR/sqa3d_eval/sqa3d_predictions_$MODEL_SIZE.jsonl}

DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-./scripts/zero2.json}
MASTER_PORT=${MASTER_PORT:-60150}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}

NUM_FRAMES=${NUM_FRAMES:-48}
FRAME_CHUNK_SIZE=${FRAME_CHUNK_SIZE:-4}
DEPTH_ENCODING=${DEPTH_ENCODING:-turbo+normals}
DEPTH_CLIP_MIN_MM=${DEPTH_CLIP_MIN_MM:-200.0}
DEPTH_CLIP_MAX_MM=${DEPTH_CLIP_MAX_MM:-10000.0}
DEPTH_NORMALS_FRAME=${DEPTH_NORMALS_FRAME:-camera}
FRAME_SAMPLING=${FRAME_SAMPLING:-pose}
POSE_SUBDIR=${POSE_SUBDIR:-pose}
POSE_MATRIX_TYPE=${POSE_MATRIX_TYPE:-c2w}

EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
SQA3D_MAX_SAMPLES=${SQA3D_MAX_SAMPLES:-}

if ! command -v deepspeed >/dev/null 2>&1; then
  echo "[ERROR] deepspeed not found in PATH. Install it (e.g., \`pip install deepspeed\`) then re-run."
  exit 1
fi

if [ -z "$VISION_ADAPTER_PATH" ] || [ ! -d "$VISION_ADAPTER_PATH" ]; then
  echo "[ERROR] VISION_ADAPTER_PATH not found: $VISION_ADAPTER_PATH"
  exit 1
fi

if [ -z "$LLM_ADAPTER_PATH" ] || [ ! -d "$LLM_ADAPTER_PATH" ]; then
  echo "[ERROR] LLM_ADAPTER_PATH not found: $LLM_ADAPTER_PATH"
  exit 1
fi

mkdir -p "$(dirname "$PRED_SAVE")"

WANDB_DISABLED=true WANDB__SERVICE_WAIT=500 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
deepspeed --num_gpus 3 --master_port "$MASTER_PORT" scripts/sqa_eval.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --local_files_only \
  --vision_adapter_path "$VISION_ADAPTER_PATH" \
  --llm_adapter_path "$LLM_ADAPTER_PATH" \
  --question_file ./data/video_instruction_tuning/sqa/llava3d_sqa3d_test_question.json \
  --gt_file ./data/video_instruction_tuning/sqa/llava3d_sqa3d_test_answer.json \
  --frames_root ./data/video_instruction_tuning/3d/frames_square \
  --color_subdir color \
  --depth_subdir depth \
  --pose_subdir "$POSE_SUBDIR" \
  --pose_matrix_type "$POSE_MATRIX_TYPE" \
  --frame_sampling "$FRAME_SAMPLING" \
  --num_frames "$NUM_FRAMES" \
  --frame_chunk_size "$FRAME_CHUNK_SIZE" \
  --depth_encoding "$DEPTH_ENCODING" \
  --depth_clip_min_mm "$DEPTH_CLIP_MIN_MM" \
  --depth_clip_max_mm "$DEPTH_CLIP_MAX_MM" \
  --depth_normals_frame "$DEPTH_NORMALS_FRAME" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --pred_save "$PRED_SAVE" \
  ${SQA3D_MAX_SAMPLES:+--max_samples "$SQA3D_MAX_SAMPLES"} \
  --use_flash_attn \
  --bf16 \
  --tf32
