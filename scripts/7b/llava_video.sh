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

MASTER_PORT_TRAIN=${MASTER_PORT_TRAIN:-62100}
MASTER_PORT_SFT=${MASTER_PORT_SFT:-62110}

DISTILL_SCOPE=${DISTILL_SCOPE:-vision}
VISION_LORA_R_LIST=${VISION_LORA_R_LIST:-"64"}
RUN_STAGE3=${RUN_STAGE3:-0}

OUT_ROOT=${OUT_ROOT:-./checkpoints/ablations/LynX_${MODEL_SIZE}/fastvideo/llava_video_178k}
REF_STATS_DIR=${REF_STATS_DIR:-./checkpoints/LynX_${MODEL_SIZE}/reference_stats}
REF_STATS_LLVAV178K=${REF_STATS_LLVAV178K:-${REF_STATS_DIR}/reference_stats_llava_video_178k.pt}

# Mapping jsons (video-id -> relative path); roots should point into academic_source/liwei_youtube_videos.
VIDEO_MAPPING_LIST=${VIDEO_MAPPING_LIST:-"./data/video_instruction_tuning/llava_video_178k/1_2_m_academic_v0_1_videos_mapping_updated.json,./data/video_instruction_tuning/llava_video_178k/1_2_m_youtube_v0_1_videos_mapping_updated.json,./data/video_instruction_tuning/llava_video_178k/2_3_m_academic_v0_1_videos_mapping_updated.json,./data/video_instruction_tuning/llava_video_178k/2_3_m_youtube_v0_1_videos_mapping_updated.json"}
VIDEO_ROOT_LIST=${VIDEO_ROOT_LIST:-"/mnt/hdd1/datasets/video_instruction_tuning/llava_video_178k/1_2_m_academic_v0_1/academic_source,/mnt/hdd1/datasets/video_instruction_tuning/llava_video_178k/1_2_m_youtube_v0_1/liwei_youtube_videos,/mnt/hdd1/datasets/video_instruction_tuning/llava_video_178k/2_3_m_academic_v0_1/academic_source,/mnt/hdd1/datasets/video_instruction_tuning/llava_video_178k/2_3_m_youtube_v0_1/liwei_youtube_videos"}

# Instruction jsons; roots should be the parent directory (the `video` field already includes academic_source/liwei_youtube_videos).
INSTRUCT_FILES=${INSTRUCT_FILES:-"./data/video_instruction_tuning/llava_video_178k/1_2_m_academic_oe_v0_1_qa_processed_2pv.json,./data/video_instruction_tuning/llava_video_178k/1_2_m_youtube_oe_v0_1_qa_processed_2pv.json,./data/video_instruction_tuning/llava_video_178k/2_3_m_academic_oe_v0_1_qa_processed_2pv.json,./data/video_instruction_tuning/llava_video_178k/2_3_m_youtube_oe_v0_1_qa_processed_2pv.json"}
INSTRUCT_VIDEO_ROOTS=${INSTRUCT_VIDEO_ROOTS:-"/mnt/hdd1/datasets/video_instruction_tuning/llava_video_178k/1_2_m_academic_v0_1,/mnt/hdd1/datasets/video_instruction_tuning/llava_video_178k/1_2_m_youtube_v0_1,/mnt/hdd1/datasets/video_instruction_tuning/llava_video_178k/2_3_m_academic_v0_1,/mnt/hdd1/datasets/video_instruction_tuning/llava_video_178k/2_3_m_youtube_v0_1"}

SLOW_NUM_FRAMES=${SLOW_NUM_FRAMES:-20}
FRAME_CHUNK_SIZE=${FRAME_CHUNK_SIZE:-4}
FAST_MULTIPLIER=${FAST_MULTIPLIER:-4}
FAST_TILE_SIZE=${FAST_TILE_SIZE:-4}
FAST_NUM_FRAMES=${FAST_NUM_FRAMES:-$((SLOW_NUM_FRAMES * FAST_MULTIPLIER))}

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

  echo "[Ablation-LLaVA-Video-178K-01] vision_lora_r=${r} -> ${exp_dir}"

  WANDB_DISABLED=true WANDB__SERVICE_WAIT=500 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  deepspeed --num_gpus "${NUM_GPUS}" --master_port "${MASTER_PORT_TRAIN}" train_lynx.py fastvideo \
    --model_name_or_path "${MODEL_PATH}" \
    --local_files_only "${LOCAL_FILES_ONLY}" \
    --output_dir "${train_dir}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --reference_stats_path "${REF_STATS_LLVAV178K}" \
    --reference_video_roots "${VIDEO_ROOT_LIST}" \
    --reference_video_mappings "${VIDEO_MAPPING_LIST}" \
    --train_video_roots "${VIDEO_ROOT_LIST}" \
    --train_video_mappings "${VIDEO_MAPPING_LIST}" \
    --reference_num_frames "${SLOW_NUM_FRAMES}" \
    --vision_frame_chunk_size "${FRAME_CHUNK_SIZE}" \
    --fast_num_frames "${FAST_NUM_FRAMES}" \
    --fast_tile_size "${FAST_TILE_SIZE}" \
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
    echo "[Ablation-LLaVA-Video-178K-01] Skipping Stage-3 SFT (RUN_STAGE3=${RUN_STAGE3})"
    continue
  fi

  WANDB_DISABLED=true WANDB__SERVICE_WAIT=500 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  deepspeed --num_gpus "${NUM_GPUS}" --master_port "${MASTER_PORT_SFT}" train_lynx_sft.py fastvideo \
    --model_name_or_path "${MODEL_PATH}" \
    --local_files_only True \
    --vision_adapter_path "${train_dir}" \
    --output_dir "${sft_dir}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --train_instruct_files "${INSTRUCT_FILES}" \
    --video_roots "${INSTRUCT_VIDEO_ROOTS}" \
    --slow_num_frames "${SLOW_NUM_FRAMES}" \
    --frame_chunk_size "${FRAME_CHUNK_SIZE}" \
    --fast_num_frames "${FAST_NUM_FRAMES}" \
    --fast_tile_size "${FAST_TILE_SIZE}" \
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
