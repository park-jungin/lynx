MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-llava-hf/llava-onevision-qwen2-7b-ov-hf}
VISION_ADAPTER_PATH=${VISION_ADAPTER_PATH:-./checkpoints/LynX_7b/audio/avqa/train}
LLM_ADAPTER_PATH=${LLM_ADAPTER_PATH:-./checkpoints/LynX_7b/audio/avqa/sft/checkpoint-6696}

OUTPUT_DIR=${OUTPUT_DIR:-./downstream_evaluation/audio}
PRED_SAVE=${PRED_SAVE:-$OUTPUT_DIR/avqa_eval/avqa_predictions_7b.json}

DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-./scripts/zero2.json}
MASTER_PORT=${MASTER_PORT:-60020}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}

EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-2}
AVQA_MAX_SAMPLES=${AVQA_MAX_SAMPLES:-}

if ! command -v deepspeed >/dev/null 2>&1; then
  echo "[ERROR] deepspeed not found in PATH. Install it (e.g., \`pip install deepspeed\`) then re-run."
  exit 1
fi

if [ -z "$VISION_ADAPTER_PATH" ] || [ ! -d "$VISION_ADAPTER_PATH" ]; then
  echo "[ERROR] VISION_ADAPTER_PATH not found: $VISION_ADAPTER_PATH"
  echo "Set it to the Stage-1/2 checkpoint dir that contains adapter_config.json + adapter_model.safetensors."
  exit 1
fi

if [ -z "$LLM_ADAPTER_PATH" ] || [ ! -d "$LLM_ADAPTER_PATH" ]; then
  echo "[ERROR] LLM_ADAPTER_PATH not found: $LLM_ADAPTER_PATH"
  echo "Set it to the Stage-3 SFT output dir (adapter folder) OR a Stage-3 checkpoint dir that contains model.safetensors."
  exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
  echo "[ERROR] OUTPUT_DIR is empty. Set OUTPUT_DIR or LLM_ADAPTER_PATH."
  exit 1
fi

mkdir -p "$(dirname "$PRED_SAVE")"

WANDB_DISABLED=true WANDB__SERVICE_WAIT=500 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
deepspeed --num_gpus 3 --master_port "$MASTER_PORT" scripts/avqa_eval.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --local_files_only \
  --vision_adapter_path "$VISION_ADAPTER_PATH" \
  --llm_adapter_path "$LLM_ADAPTER_PATH" \
  --annotation_file ./data/video_instruction_tuning/avqa/val_qa.json \
  --video_root ./data/video_instruction_tuning/avqa/videos \
  --video_mapping ./data/video_instruction_tuning/avqa/avqa_from_vid_to_video_name.json \
  --num_frames 10 \
  --frame_chunk_size 4 \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --pred_save "$PRED_SAVE" \
  ${AVQA_MAX_SAMPLES:+--max_samples "$AVQA_MAX_SAMPLES"} \
  --pool_video_tokens \
  --use_flash_attn \
  --bf16 \
  --tf32
