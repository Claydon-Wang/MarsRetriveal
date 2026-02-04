#!/usr/bin/env bash
# Usage: bash scripts/landform_retrieval/openclip.sh 0,1,2,3 (or export CUDA_VISIBLE_DEVICES beforehand)
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH

# Huggingface settings
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/sharedata/ssd_large/common/VLMs/
export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/VLMs/datasets/

TASK_CONFIG=LandformRetrieval
MODEL_CONFIG=OpenCLIP
EXP_NAME=main_exp

# query settings (landform retrieval uses dataset class list)
QUERY_MODE=text

# model (OpenCLIP by default)
MODEL_NAMES=(
  PE-Core-L-14-336
)
PRETRAINEDS=(
  hf-hub:timm/PE-Core-L-14-336
)

IMAGE_ENCODER_TYPE=openclip
TEXT_ENCODER_TYPE=openclip

NPROC=$(( $(echo "${CUDA_VISIBLE_DEVICES:-}" | tr -cd ',' | wc -c) + 1 ))

PROMPT_COUNTS=(1 3 5 7 10)

for IDX in "${!MODEL_NAMES[@]}"; do
  MODEL_NAME="${MODEL_NAMES[$IDX]}"
  PRETRAINED="${PRETRAINEDS[$IDX]}"

  # optional distributed DB build (skips if DB already exists)
  if [[ "${NPROC}" -gt 1 ]]; then
    torchrun --nproc_per_node=${NPROC} build_db.py \
      --task_config "${TASK_CONFIG}" \
      --model_config "${MODEL_CONFIG}" \
      --exp_name "${EXP_NAME}" \
      --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
      --text_encoder_type "none" \
      --model_name "${MODEL_NAME}" \
      --pretrained "${PRETRAINED}"
  fi

  # run retrieval (prompt ablation)
  for PROMPT_COUNT in "${PROMPT_COUNTS[@]}"; do
    python main.py \
      --task_config "${TASK_CONFIG}" \
      --model_config "${MODEL_CONFIG}" \
      --exp_name "${EXP_NAME}_pc${PROMPT_COUNT}" \
      --query_mode "${QUERY_MODE}" \
      --model_name "${MODEL_NAME}" \
      --pretrained "${PRETRAINED}" \
      --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
      --text_encoder_type "${TEXT_ENCODER_TYPE}" \
      --prompt_count "${PROMPT_COUNT}"
  done
done
