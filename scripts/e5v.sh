#!/usr/bin/env bash
# Usage: bash scripts/e5v.sh 0,1  (or export CUDA_VISIBLE_DEVICES beforehand)
export TORCH_DISTRIBUTED_TIMEOUT=86400
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH

# Hugging Face mirrors/caches (adjust to your environment)
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/mnt/sharedata/ssd_large/common/VLMs/}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/mnt/sharedata/ssd_large/common/VLMs/datasets/}

CONFIG_NAME=MarsRetrievalE5V
EXP_NAME=main_exp

# Query settings
QUERY_MODE=text   # image | text | hybrid
QUERY_TEXT=yardangs
QUERY_IMAGES=/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_localization/image_queries/${QUERY_TEXT}
GROUND_TRUTH_CSV=/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_localization/dataset/ground_truth/${QUERY_TEXT}.csv

# e5-V model (LLaVA-Next backbone)
MODEL_NAME=royokong/e5-v
PRETRAINED=e5-v
IMAGE_ENCODER_TYPE=e5-v
TEXT_ENCODER_TYPE=e5-v

# Optional distributed DB build (skips if DB already exists)
NPROC=$(( $(echo "${CUDA_VISIBLE_DEVICES:-}" | tr -cd ',' | wc -c) + 1 ))
if [[ "${NPROC}" -gt 1 ]]; then
  echo "Building DB with torchrun on ${NPROC} GPUs: ${CUDA_VISIBLE_DEVICES}"
  torchrun --nproc_per_node=${NPROC} build_db.py \
    --config_name "${CONFIG_NAME}" \
    --exp_name "${EXP_NAME}" \
    --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
    --text_encoder_type "none" \
    --model_name "${MODEL_NAME}" \
    --pretrained "${PRETRAINED}"
fi

# Run retrieval
python main.py \
  --config_name "${CONFIG_NAME}" \
  --exp_name "${EXP_NAME}" \
  --query_mode "${QUERY_MODE}" \
  --query_text "${QUERY_TEXT}" \
  --query_images ${QUERY_IMAGES} \
  --ground_truth_csv "${GROUND_TRUTH_CSV}" \
  --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
  --text_encoder_type "${TEXT_ENCODER_TYPE}" \
  --model_name "${MODEL_NAME}" \
  --pretrained "${PRETRAINED}"
