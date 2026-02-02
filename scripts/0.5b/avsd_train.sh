MODEL_SIZE=${MODEL_SIZE:-0.5b}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/LynX_$MODEL_SIZE/audio/avsd/train}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-./scripts/zero2.json}
MASTER_PORT=${MASTER_PORT:-60030}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
DISTILL_SCOPE=${DISTILL_SCOPE:-vision}
MODEL_PATH=${MODEL_PATH:-llava-hf/llava-onevision-qwen2-$MODEL_SIZE-ov-hf}

# AVSD (Charades) paths
AVSD_TRAIN_INSTRUCT_FILE=${AVSD_TRAIN_INSTRUCT_FILE:-./data/video_instruction_tuning/avsd/avsd_train_instruct.json}
AVSD_TRAIN_VIDEO_ROOT=${AVSD_TRAIN_VIDEO_ROOT:-./data/video_instruction_tuning/avsd/Charades_v1}
AVSD_TRAIN_VIDEO_MAPPING=${AVSD_TRAIN_VIDEO_MAPPING:-./data/video_instruction_tuning/avsd/avsd_all_videos_mapping.json}

AVSD_EVAL_VIDEO_ROOT=${AVSD_EVAL_VIDEO_ROOT:-./data/video_instruction_tuning/avsd/Charades_vu17_test}
AVSD_EVAL_GT_FILE=${AVSD_EVAL_GT_FILE:-./data/video_instruction_tuning/avsd/avsd_coco_version_test_gt.json}
# Frame sampling policy:
# - AVSD average video length is >10s, so default fps=1.
# - Our code uses a fixed `num_frames` sampled uniformly across the video.
AVSD_FPS=${AVSD_FPS:-1}
AVSD_NUM_FRAMES=${AVSD_NUM_FRAMES:-30}

# ImageBind-style mel clips: clip_stride_s should match fps (fps=2 -> 0.5, fps=1 -> 1.0).
AVSD_AUDIO_CLIP_STRIDE_S=${AVSD_AUDIO_CLIP_STRIDE_S:-1.0}

# LLaVA-OV video feature chunking
AVSD_FRAME_CHUNK_SIZE=${AVSD_FRAME_CHUNK_SIZE:-4}



if ! command -v deepspeed >/dev/null 2>&1; then
  echo "[ERROR] deepspeed not found in PATH. Install it (e.g., \`pip install deepspeed\`) then re-run."
  exit 1
fi

WANDB_DISABLED=true WANDB__SERVICE_WAIT=500 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
deepspeed --num_gpus 3 --master_port "$MASTER_PORT" train_lynx.py \
  --model_name_or_path "$MODEL_PATH" \
  --local_files_only True \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --reference_video_root "$AVSD_TRAIN_VIDEO_ROOT" \
  --reference_video_mapping "$AVSD_TRAIN_VIDEO_MAPPING" \
  --reference_num_frames "$AVSD_NUM_FRAMES" \
  --reference_stats_path "$OUTPUT_DIR/reference_stats.pt" \
  --train_video_root "$AVSD_TRAIN_VIDEO_ROOT" \
  --train_video_mapping "$AVSD_TRAIN_VIDEO_MAPPING" \
  --train_annotation_file "./data/video_instruction_tuning/avsd/avsd_train_instruct.json" \
  --vision_frame_chunk_size "$AVSD_FRAME_CHUNK_SIZE" \
  --lambda_distill 1.0 \
  --distill_scope "$DISTILL_SCOPE" \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 2 \
  --num_train_epochs 10 \
  --report_to none \
  --disable_tqdm False \
  --bf16 True \
  --tf32 True \
  --remove_unused_columns False \
  --resume False
