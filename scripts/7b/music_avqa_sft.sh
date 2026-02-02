MODEL_SIZE=${MODEL_SIZE:-7b}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/LynX_$MODEL_SIZE/audio/music_avqa/sft}
VISION_ADAPTER_PATH=${VISION_ADAPTER_PATH:-./checkpoints/LynX_$MODEL_SIZE/audio/music_avqa/train}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-./scripts/zero2.json}
MASTER_PORT=${MASTER_PORT:-60060}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
MODEL_PATH=${MODEL_PATH:-llava-hf/llava-onevision-qwen2-$MODEL_SIZE-ov-hf}

DATASET_NAME=${DATASET_NAME:-music_avqa}

# Music-AVQA paths
MUSIC_AVQA_TRAIN_INSTRUCT_FILE=${MUSIC_AVQA_TRAIN_INSTRUCT_FILE:-./data/video_instruction_tuning/music_avqa/music_avqa_train_instruct_duplicate_audio.json}
MUSIC_AVQA_VIDEO_ROOT=${MUSIC_AVQA_VIDEO_ROOT:-./data/video_instruction_tuning/music_avqa}
MUSIC_AVQA_VIDEO_MAPPING=${MUSIC_AVQA_VIDEO_MAPPING:-./data/video_instruction_tuning/music_avqa/music_avqa_all_videos_mapping.json}
MUSIC_AVQA_EVAL_ANN_FILE=${MUSIC_AVQA_EVAL_ANN_FILE:-./data/video_instruction_tuning/music_avqa/music_avqa_updated_avqa-test.json}

# Frame sampling policy:
# - Music-AVQA videos are long (>10s), so default fps=1.
# - Our code uses a fixed `num_frames` sampled uniformly across the video.
MUSIC_AVQA_FPS=${MUSIC_AVQA_FPS:-1}
MUSIC_AVQA_NUM_FRAMES=${MUSIC_AVQA_NUM_FRAMES:-32}

# ImageBind-style mel clips: clip_stride_s should match fps (fps=2 -> 0.5, fps=1 -> 1.0).
MUSIC_AVQA_AUDIO_CLIP_STRIDE_S=${MUSIC_AVQA_AUDIO_CLIP_STRIDE_S:-1.0}

# LLaVA-OV video feature chunking
MUSIC_AVQA_FRAME_CHUNK_SIZE=${MUSIC_AVQA_FRAME_CHUNK_SIZE:-4}

if ! command -v deepspeed >/dev/null 2>&1; then
  echo "[ERROR] deepspeed not found in PATH. Install it (e.g., \`pip install deepspeed\`) then re-run."
  exit 1
fi

if [ ! -d "$VISION_ADAPTER_PATH" ]; then
  echo "[ERROR] VISION_ADAPTER_PATH not found: $VISION_ADAPTER_PATH"
  echo "Set it to a Stage-1/2 checkpoint directory that contains the vision LoRA adapter."
  exit 1
fi

WANDB_DISABLED=true WANDB__SERVICE_WAIT=500 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
deepspeed --num_gpus 3 --master_port "$MASTER_PORT" train_lynx_sft.py \
  --model_name_or_path "$MODEL_PATH" \
  --local_files_only True \
  --vision_adapter_path "$VISION_ADAPTER_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --dataset music_avqa \
  --train_instruct_file "$MUSIC_AVQA_TRAIN_INSTRUCT_FILE" \
  --video_root "$MUSIC_AVQA_VIDEO_ROOT" \
  --video_mapping "$MUSIC_AVQA_VIDEO_MAPPING" \
  --num_frames "$MUSIC_AVQA_NUM_FRAMES" \
  --frame_chunk_size "$MUSIC_AVQA_FRAME_CHUNK_SIZE" \
  --audio_clip_stride_s "$MUSIC_AVQA_AUDIO_CLIP_STRIDE_S" \
  --enable_music_avqa_eval True \
  --music_avqa_annotation_file "$MUSIC_AVQA_EVAL_ANN_FILE" \
  --music_avqa_video_root "$MUSIC_AVQA_VIDEO_ROOT" \
  --music_avqa_video_mapping "$MUSIC_AVQA_VIDEO_MAPPING" \
  --music_avqa_num_frames "$MUSIC_AVQA_NUM_FRAMES" \
  --music_avqa_eval_batch_size 8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --eval_strategy steps \
  --eval_steps 500 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 2 \
  --num_train_epochs 2 \
  --report_to none \
  --disable_tqdm False \
  --bf16 True \
  --tf32 True \
  --remove_unused_columns False
