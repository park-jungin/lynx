MODEL_SIZE=${MODEL_SIZE:-7b}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/LynX_$MODEL_SIZE/3d/scanqa/sft}
VISION_ADAPTER_PATH=${VISION_ADAPTER_PATH:-./checkpoints/LynX_$MODEL_SIZE/3d/scanqa/train}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-./scripts/zero2.json}
MASTER_PORT=${MASTER_PORT:-60120}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
MODEL_PATH=${MODEL_PATH:-llava-hf/llava-onevision-qwen2-$MODEL_SIZE-ov-hf}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-True}

NUM_FRAMES=${NUM_FRAMES:-10}
FRAME_CHUNK_SIZE=${FRAME_CHUNK_SIZE:-4}

DEPTH_ENCODING=${DEPTH_ENCODING:-turbo+normals}
DEPTH_CLIP_MIN_MM=${DEPTH_CLIP_MIN_MM:-200.0}
DEPTH_CLIP_MAX_MM=${DEPTH_CLIP_MAX_MM:-10000.0}

if ! command -v deepspeed >/dev/null 2>&1; then
  echo "[ERROR] deepspeed not found in PATH. Install it (e.g., \`pip install deepspeed\`) then re-run."
  exit 1
fi

if [ ! -d "$VISION_ADAPTER_PATH" ]; then
  echo "[ERROR] VISION_ADAPTER_PATH not found: $VISION_ADAPTER_PATH"
  echo "Set it to a Stage-1/2 checkpoint directory that contains the depth vision-LoRA adapter."
  exit 1
fi

python - <<'PY'
import importlib
import shutil

missing = []
for mod in ["pycocoevalcap"]:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)

if missing:
    print(f"[WARN] ScanQA evaluation metrics need: {', '.join(missing)} (pip install pycocoevalcap).")
if shutil.which("java") is None:
    print("[WARN] `java` not found in PATH. METEOR/SPICE (pycocoevalcap) may fail and metrics may be skipped.")
PY

WANDB_DISABLED=true WANDB__SERVICE_WAIT=500 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
deepspeed --num_gpus 3 --master_port "$MASTER_PORT" train_lynx_sft.py 3d \
  --model_name_or_path "$MODEL_PATH" \
  --local_files_only "$LOCAL_FILES_ONLY" \
  --vision_adapter_path "$VISION_ADAPTER_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --dataset scanqa \
  --train_instruct_file ./data/video_instruction_tuning/scannet/scanqa_train_instruct.json \
  --frames_root ./data/video_instruction_tuning/3d/frames_square \
  --color_subdir color \
  --depth_subdir depth \
  --num_frames "$NUM_FRAMES" \
  --frame_chunk_size "$FRAME_CHUNK_SIZE" \
  --depth_encoding "$DEPTH_ENCODING" \
  --depth_clip_min_mm "$DEPTH_CLIP_MIN_MM" \
  --depth_clip_max_mm "$DEPTH_CLIP_MAX_MM" \
  --enable_scanqa_eval True \
  --scanqa_question_file ./data/video_instruction_tuning/scannet/llava-3d-scanqa_val_question.json \
  --scanqa_gt_file ./data/video_instruction_tuning/scannet/llava3d_scanqa_val_answer.json \
  --scanqa_num_frames "$NUM_FRAMES" \
  --scanqa_eval_batch_size 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --eval_strategy steps \
  --eval_steps 500 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 2 \
  --num_train_epochs 2 \
  --report_to none \
  --disable_tqdm False \
  --bf16 True \
  --tf32 True \
  --remove_unused_columns False
