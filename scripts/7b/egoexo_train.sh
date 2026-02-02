#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

MODEL_SIZE=${MODEL_SIZE:-7b}
MODEL_PATH=${MODEL_PATH:-llava-hf/llava-onevision-qwen2-${MODEL_SIZE}-ov-hf}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-True}

NUM_GPUS=${NUM_GPUS:-3}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-./scripts/zero2.json}

MASTER_PORT_TRAIN=${MASTER_PORT_TRAIN:-62200}
MASTER_PORT_SFT=${MASTER_PORT_SFT:-62210}

DISTILL_SCOPE=${DISTILL_SCOPE:-vision}
VISION_LORA_R_LIST=${VISION_LORA_R_LIST:-"64"}
RUN_STAGE3=${RUN_STAGE3:-0}

DATASET_NAME=${DATASET_NAME:-egoexo}

EGOEXO_VIDEO_ROOT=${EGOEXO_VIDEO_ROOT:-${DATA_ROOT:-./data/video_instruction_tuning/egoexo}}
EGOEXO_TAKE_TO_VIDEO_MAP=${EGOEXO_TAKE_TO_VIDEO_MAP:-${TAKE_TO_VIDEO_MAP:-${EGOEXO_VIDEO_ROOT}/from_take_id_to_video.json}}
EGOEXO_TRAIN_INSTRUCT_FILE=${EGOEXO_TRAIN_INSTRUCT_FILE:-${TRAIN_INSTRUCT_FILE:-${EGOEXO_VIDEO_ROOT}/proficiency_demonstrator_train_instruct.json}}

# Backward-compatible aliases
DATA_ROOT=${DATA_ROOT:-${EGOEXO_VIDEO_ROOT}}
TAKE_TO_VIDEO_MAP=${TAKE_TO_VIDEO_MAP:-${EGOEXO_TAKE_TO_VIDEO_MAP}}
TRAIN_INSTRUCT_FILE=${TRAIN_INSTRUCT_FILE:-${EGOEXO_TRAIN_INSTRUCT_FILE}}

OUT_ROOT=${OUT_ROOT:-./checkpoints/ablations/LynX_${MODEL_SIZE}/egoexo}
REF_STATS_DIR=${REF_STATS_DIR:-./checkpoints/LynX_${MODEL_SIZE}/reference_stats}
REF_STATS_EGOEXO=${REF_STATS_EGOEXO:-${REF_STATS_DIR}/reference_stats_egoexo.pt}

EGOEXO_NUM_FRAMES=${EGOEXO_NUM_FRAMES:-${NUM_FRAMES:-20}}
EGOEXO_FRAME_CHUNK_SIZE=${EGOEXO_FRAME_CHUNK_SIZE:-${FRAME_CHUNK_SIZE:-4}}

# Backward-compatible aliases
NUM_FRAMES=${NUM_FRAMES:-${EGOEXO_NUM_FRAMES}}
FRAME_CHUNK_SIZE=${FRAME_CHUNK_SIZE:-${EGOEXO_FRAME_CHUNK_SIZE}}

NUM_EPOCHS_TRAIN=${NUM_EPOCHS_TRAIN:-10}
LR_TRAIN=${LR_TRAIN:-2e-5}
TRAIN_BATCH=${TRAIN_BATCH:-8}
GRAD_ACCUM=${GRAD_ACCUM:-4}

SFT_LR=${SFT_LR:-5e-4}
SFT_BATCH=${SFT_BATCH:-1}
SFT_GRAD_ACCUM=${SFT_GRAD_ACCUM:-4}
SFT_EPOCHS=${SFT_EPOCHS:-1}

if ! command -v deepspeed >/dev/null 2>&1; then
  echo "[ERROR] deepspeed not found in PATH."
  exit 1
fi

mkdir -p "${REF_STATS_DIR}"

for r in ${VISION_LORA_R_LIST}; do
  exp_dir="${OUT_ROOT}/abl_01_vision_lora_r${r}"
  train_dir="${exp_dir}/train"
  sft_dir="${exp_dir}/sft"
  mkdir -p "${train_dir}" "${sft_dir}"

  echo "[EgoExo-01] vision_lora_r=${r} -> ${exp_dir}"

  WANDB_DISABLED=true WANDB__SERVICE_WAIT=500 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  deepspeed --num_gpus "${NUM_GPUS}" --master_port "${MASTER_PORT_TRAIN}" train_lynx.py egoexo \
    --model_name_or_path "${MODEL_PATH}" \
    --local_files_only "${LOCAL_FILES_ONLY}" \
    --output_dir "${train_dir}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --reference_video_root "${DATA_ROOT}" \
    --reference_video_mapping "${TAKE_TO_VIDEO_MAP}" \
    --reference_stats_path "${REF_STATS_EGOEXO}" \
    --train_video_root "${DATA_ROOT}" \
    --train_video_mapping "${TAKE_TO_VIDEO_MAP}" \
    --reference_num_frames "${NUM_FRAMES}" \
    --train_num_frames "${NUM_FRAMES}" \
    --vision_frame_chunk_size "${FRAME_CHUNK_SIZE}" \
    --lambda_distill 1.0 \
    --distill_scope "${DISTILL_SCOPE}" \
    --vision_lora_r "${r}" \
    --vision_lora_alpha "$((r * 2))" \
    --per_device_train_batch_size "${TRAIN_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LR_TRAIN}" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 10 \
    --num_train_epochs "${NUM_EPOCHS_TRAIN}" \
    --report_to none \
    --disable_tqdm False \
    --bf16 True \
    --tf32 True \
    --remove_unused_columns False

  if [ "${RUN_STAGE3}" != "1" ]; then
    echo "[EgoExo-01] Skipping Stage-3 SFT (RUN_STAGE3=${RUN_STAGE3})"
    continue
  fi

  WANDB_DISABLED=true WANDB__SERVICE_WAIT=500 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  deepspeed --num_gpus "${NUM_GPUS}" --master_port "${MASTER_PORT_SFT}" train_lynx_sft.py egoexo \
    --model_name_or_path "${MODEL_PATH}" \
    --local_files_only True \
    --vision_adapter_path "${train_dir}" \
    --output_dir "${sft_dir}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --train_instruct_file "${TRAIN_INSTRUCT_FILE}" \
    --video_root "${DATA_ROOT}" \
    --take_id_to_video_mapping "${TAKE_TO_VIDEO_MAP}" \
    --num_frames "${NUM_FRAMES}" \
    --frame_chunk_size "${FRAME_CHUNK_SIZE}" \
    --per_device_train_batch_size "${SFT_BATCH}" \
    --gradient_accumulation_steps "${SFT_GRAD_ACCUM}" \
    --learning_rate "${SFT_LR}" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 2 \
    --num_train_epochs "${SFT_EPOCHS}" \
    --report_to none \
    --disable_tqdm False \
    --bf16 True \
    --tf32 True \
    --remove_unused_columns False
done
