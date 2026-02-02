MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}
VISION_ADAPTER_PATH=${VISION_ADAPTER_PATH:-./checkpoints/LynX_0.5b/audio/music_avqa/train}
LLM_ADAPTER_PATH=${LLM_ADAPTER_PATH:-./checkpoints/LynX_0.5b/audio/music_avqa/sft}

OUTPUT_DIR=${OUTPUT_DIR:-./downstream_evaluation/audio}
PRED_SAVE=${PRED_SAVE:-$OUTPUT_DIR/music_avqa_eval/music_avqa_predictions_0.5b.json}

DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-./scripts/zero2.json}
MASTER_PORT=${MASTER_PORT:-60081}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}

EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
MUSIC_AVQA_MAX_SAMPLES=${MUSIC_AVQA_MAX_SAMPLES:-}

DATASET_NAME=${DATASET_NAME:-music_avqa}

# Music-AVQA paths
MUSIC_AVQA_TRAIN_INSTRUCT_FILE=${MUSIC_AVQA_TRAIN_INSTRUCT_FILE:-./data/video_instruction_tuning/music_avqa/music_avqa_train_instruct_duplicate_audio.json}
MUSIC_AVQA_VIDEO_ROOT=${MUSIC_AVQA_VIDEO_ROOT:-./data/video_instruction_tuning/music_avqa}
MUSIC_AVQA_VIDEO_MAPPING=${MUSIC_AVQA_VIDEO_MAPPING:-./data/video_instruction_tuning/music_avqa/music_avqa_all_videos_mapping.json}
MUSIC_AVQA_EVAL_ANN_FILE=${MUSIC_AVQA_EVAL_ANN_FILE:-./data/video_instruction_tuning/music_avqa/music_avqa_updated_avqa-test.json}

MUSIC_AVQA_FPS=${MUSIC_AVQA_FPS:-1}
MUSIC_AVQA_NUM_FRAMES=${MUSIC_AVQA_NUM_FRAMES:-32}
MUSIC_AVQA_AUDIO_CLIP_STRIDE_S=${MUSIC_AVQA_AUDIO_CLIP_STRIDE_S:-1.0}
MUSIC_AVQA_FRAME_CHUNK_SIZE=${MUSIC_AVQA_FRAME_CHUNK_SIZE:-4}

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

mkdir -p "$(dirname "$PRED_SAVE")"

WANDB_DISABLED=true WANDB__SERVICE_WAIT=500 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
deepspeed --num_gpus 3 --master_port "$MASTER_PORT" scripts/music_avqa_eval.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --local_files_only \
  --vision_adapter_path "$VISION_ADAPTER_PATH" \
  --llm_adapter_path "$LLM_ADAPTER_PATH" \
  --annotation_file "$MUSIC_AVQA_EVAL_ANN_FILE" \
  --video_root "$MUSIC_AVQA_VIDEO_ROOT" \
  --video_mapping "$MUSIC_AVQA_VIDEO_MAPPING" \
  --num_frames "$MUSIC_AVQA_NUM_FRAMES" \
  --frame_chunk_size "$MUSIC_AVQA_FRAME_CHUNK_SIZE" \
  --audio_clip_stride_s "$MUSIC_AVQA_AUDIO_CLIP_STRIDE_S" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --pred_save "$PRED_SAVE" \
  ${MUSIC_AVQA_MAX_SAMPLES:+--max_samples "$MUSIC_AVQA_MAX_SAMPLES"} \
  --use_flash_attn \
  --bf16 \
  --tf32

python tools/audio/music-avqa/calculate_acc.py \
  --prediction-path "$PRED_SAVE" \
  --annotation-path "$MUSIC_AVQA_EVAL_ANN_FILE"

