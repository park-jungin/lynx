MODEL_SIZE=${MODEL_SIZE:-0.5b}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/LynX_$MODEL_SIZE/audio/avqa/train}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-./scripts/zero2.json}
MASTER_PORT=${MASTER_PORT:-60000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
DISTILL_SCOPE=${DISTILL_SCOPE:-vision}
MODEL_PATH=${MODEL_PATH:-llava-hf/llava-onevision-qwen2-$MODEL_SIZE-ov-hf}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-True}
SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL:-}


if ! command -v deepspeed >/dev/null 2>&1; then
  echo "[ERROR] deepspeed not found in PATH. Install it (e.g., \`pip install deepspeed\`) then re-run."
  exit 1
fi

if [ -z "${SAVE_ONLY_MODEL}" ]; then
  if [ "${MODEL_SIZE}" = "7b" ] || [ "${MODEL_SIZE}" = "7B" ]; then
    SAVE_ONLY_MODEL=True
  else
    SAVE_ONLY_MODEL=False
  fi
fi

WANDB_DISABLED=true WANDB__SERVICE_WAIT=500 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
deepspeed --num_gpus 3 --master_port "$MASTER_PORT" train_lynx.py \
  --model_name_or_path "$MODEL_PATH" \
  --local_files_only "$LOCAL_FILES_ONLY" \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --reference_video_root ./data/video_instruction_tuning/avqa/videos \
  --reference_video_mapping ./data/video_instruction_tuning/avqa/avqa_from_vid_to_video_name.json \
  --train_video_root ./data/video_instruction_tuning/avqa/videos \
  --train_video_mapping ./data/video_instruction_tuning/avqa/avqa_from_vid_to_video_name.json \
  --train_annotation_file ./data/video_instruction_tuning/avqa/train_qa.json \
  --avqa_annotation_file ./data/video_instruction_tuning/avqa/val_qa.json \
  --avqa_eval_batch_size 8 \
  --reference_num_frames 20 \
  --lambda_distill 1.0 \
  --distill_scope "$DISTILL_SCOPE" \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 10 \
  --save_only_model "$SAVE_ONLY_MODEL" \
  --num_train_epochs 2 \
  --report_to none \
  --disable_tqdm False \
  --bf16 True \
  --tf32 True \
  --remove_unused_columns False \
  --resume False