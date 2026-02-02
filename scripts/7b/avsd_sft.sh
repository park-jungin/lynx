MODEL_SIZE=${MODEL_SIZE:-7b}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/LynX_$MODEL_SIZE/audio/avsd/sft}
VISION_ADAPTER_PATH=${VISION_ADAPTER_PATH:-./checkpoints/LynX_$MODEL_SIZE/audio/avsd/train}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-./scripts/zero2.json}
MASTER_PORT=${MASTER_PORT:-60050}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
MODEL_PATH=${MODEL_PATH:-llava-hf/llava-onevision-qwen2-$MODEL_SIZE-ov-hf}

DATASET_NAME=${DATASET_NAME:-avsd}

# AVSD (Charades) paths
AVSD_TRAIN_INSTRUCT_FILE=${AVSD_TRAIN_INSTRUCT_FILE:-./data/video_instruction_tuning/avsd/avsd_train_instruct.json}
AVSD_TRAIN_VIDEO_ROOT=${AVSD_TRAIN_VIDEO_ROOT:-./data/video_instruction_tuning/avsd/Charades_v1}
AVSD_TRAIN_VIDEO_MAPPING=${AVSD_TRAIN_VIDEO_MAPPING:-./data/video_instruction_tuning/avsd/avsd_all_videos_mapping.json}

AVSD_EVAL_VIDEO_ROOT=${AVSD_EVAL_VIDEO_ROOT:-./data/video_instruction_tuning/avsd/Charades_vu17_test}
AVSD_EVAL_GT_FILE=${AVSD_EVAL_GT_FILE:-./data/video_instruction_tuning/avsd/avsd_coco_version_test_gt.json}
AVSD_EVAL_ANN_FILE=${AVSD_EVAL_ANN_FILE:-./data/video_instruction_tuning/avsd/mock_test_set4DSTC10-AVSD_from_DSTC7_singref.json}

# Frame sampling policy:
# - AVSD average video length is >10s, so default fps=1.
# - Our code uses a fixed `num_frames` sampled uniformly across the video.
AVSD_FPS=${AVSD_FPS:-1}
AVSD_NUM_FRAMES=${AVSD_NUM_FRAMES:-20}

# ImageBind-style mel clips: clip_stride_s should match fps (fps=2 -> 0.5, fps=1 -> 1.0).
AVSD_AUDIO_CLIP_STRIDE_S=${AVSD_AUDIO_CLIP_STRIDE_S:-1.0}

# LLaVA-OV video feature chunking
AVSD_FRAME_CHUNK_SIZE=${AVSD_FRAME_CHUNK_SIZE:-4}

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
  --dataset avsd \
  --train_instruct_file "$AVSD_TRAIN_INSTRUCT_FILE" \
  --video_root "$AVSD_TRAIN_VIDEO_ROOT" \
  --video_mapping "$AVSD_TRAIN_VIDEO_MAPPING" \
  --num_frames "$AVSD_NUM_FRAMES" \
  --frame_chunk_size "$AVSD_FRAME_CHUNK_SIZE" \
  --audio_clip_stride_s "$AVSD_AUDIO_CLIP_STRIDE_S" \
  --enable_avsd_eval True \
  --avsd_annotation_file "$AVSD_EVAL_ANN_FILE" \
  --avsd_gt_file "$AVSD_EVAL_GT_FILE" \
  --avsd_video_root "$AVSD_EVAL_VIDEO_ROOT" \
  --avsd_num_frames "$AVSD_NUM_FRAMES" \
  --avsd_frame_chunk_size "$AVSD_FRAME_CHUNK_SIZE" \
  --avsd_eval_batch_size 8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --eval_strategy steps \
  --eval_steps 100 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 2 \
  --num_train_epochs 1 \
  --report_to none \
  --disable_tqdm False \
  --bf16 True \
  --tf32 True \
  --remove_unused_columns False
