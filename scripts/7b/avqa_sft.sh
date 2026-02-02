MODEL_SIZE=${MODEL_SIZE:-7b}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/LynX_$MODEL_SIZE/audio/avqa/sft}
VISION_ADAPTER_PATH=${VISION_ADAPTER_PATH:-./checkpoints/LynX_$MODEL_SIZE/audio/avqa/train/}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-./scripts/zero2.json}
MASTER_PORT=${MASTER_PORT:-60010}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
MODEL_PATH=${MODEL_PATH:-llava-hf/llava-onevision-qwen2-$MODEL_SIZE-ov-hf}

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
  --train_instruct_file ./data/video_instruction_tuning/avqa/avqa_train_qa_instruct.json \
  --video_root ./data/video_instruction_tuning/avqa/videos \
  --video_mapping ./data/video_instruction_tuning/avqa/avqa_from_vid_to_video_name.json \
  --num_frames 10 \
  --frame_chunk_size 4 \
  --enable_avqa_eval True \
  --avqa_annotation_file ./data/video_instruction_tuning/avqa/val_qa.json \
  --avqa_video_root ./data/video_instruction_tuning/avqa/videos \
  --avqa_video_mapping ./data/video_instruction_tuning/avqa/avqa_from_vid_to_video_name.json \
  --avqa_num_frames 10 \
  --avqa_eval_batch_size 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --eval_strategy steps \
  --eval_steps 2000 \
  --save_strategy steps \
  --save_steps 4000 \
  --save_total_limit 2 \
  --num_train_epochs 2 \
  --report_to none \
  --disable_tqdm False \
  --bf16 True \
  --tf32 True \
  --remove_unused_columns False
