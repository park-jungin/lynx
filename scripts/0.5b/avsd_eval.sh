MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}
VISION_ADAPTER_PATH=${VISION_ADAPTER_PATH:-./checkpoints/LynX_0.5b/audio/avsd/train}
LLM_ADAPTER_PATH=${LLM_ADAPTER_PATH:-./checkpoints/LynX_0.5b/audio/avsd/sft}

OUTPUT_DIR=${OUTPUT_DIR:-./downstream_evaluation/audio}
PRED_SAVE=${PRED_SAVE:-$OUTPUT_DIR/avsd_eval/avsd_predictions_0.5b.json}

DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-./scripts/zero2.json}
MASTER_PORT=${MASTER_PORT:-60071}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}

EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
AVSD_MAX_SAMPLES=${AVSD_MAX_SAMPLES:-}

DATASET_NAME=${DATASET_NAME:-avsd}

# AVSD (Charades) paths
AVSD_TRAIN_INSTRUCT_FILE=${AVSD_TRAIN_INSTRUCT_FILE:-./data/video_instruction_tuning/avsd/avsd_train_instruct.json}
AVSD_TRAIN_VIDEO_ROOT=${AVSD_TRAIN_VIDEO_ROOT:-./data/video_instruction_tuning/avsd/Charades_v1}
AVSD_TRAIN_VIDEO_MAPPING=${AVSD_TRAIN_VIDEO_MAPPING:-./data/video_instruction_tuning/avsd/avsd_all_videos_mapping.json}

AVSD_EVAL_VIDEO_ROOT=${AVSD_EVAL_VIDEO_ROOT:-./data/video_instruction_tuning/avsd/Charades_vu17_test}
AVSD_EVAL_GT_FILE=${AVSD_EVAL_GT_FILE:-./data/video_instruction_tuning/avsd/avsd_coco_version_test_gt.json}
AVSD_EVAL_ANN_FILE=${AVSD_EVAL_ANN_FILE:-./data/video_instruction_tuning/avsd/mock_test_set4DSTC10-AVSD_from_DSTC7_singref.json}

AVSD_FPS=${AVSD_FPS:-1}
AVSD_NUM_FRAMES=${AVSD_NUM_FRAMES:-20}
AVSD_AUDIO_CLIP_STRIDE_S=${AVSD_AUDIO_CLIP_STRIDE_S:-1.0}
AVSD_FRAME_CHUNK_SIZE=${AVSD_FRAME_CHUNK_SIZE:-4}

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
deepspeed --num_gpus 3 --master_port "$MASTER_PORT" scripts/avsd_eval.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --local_files_only \
  --vision_adapter_path "$VISION_ADAPTER_PATH" \
  --llm_adapter_path "$LLM_ADAPTER_PATH" \
  --annotation_file "$AVSD_EVAL_ANN_FILE" \
  --gt_file "$AVSD_EVAL_GT_FILE" \
  --video_root "$AVSD_EVAL_VIDEO_ROOT" \
  --num_frames "$AVSD_NUM_FRAMES" \
  --frame_chunk_size "$AVSD_FRAME_CHUNK_SIZE" \
  --audio_clip_stride_s "$AVSD_AUDIO_CLIP_STRIDE_S" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --pred_save "$PRED_SAVE" \
  ${AVSD_MAX_SAMPLES:+--max_samples "$AVSD_MAX_SAMPLES"} \
  --use_flash_attn \
  --bf16 \
  --tf32

python tools/audio/avsd/run_coco_eval.py \
  --gt-file "$AVSD_EVAL_GT_FILE" \
  --results-file "$PRED_SAVE"

