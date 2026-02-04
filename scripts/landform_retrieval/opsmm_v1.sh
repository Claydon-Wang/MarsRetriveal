#!/usr/bin/env bash
# Usage: bash scripts/landform_retrieval/opsmm_v1.sh 0,1 (or export CUDA_VISIBLE_DEVICES beforehand)
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH

# Hugging Face mirrors/caches (adjust to your environment)
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/sharedata/ssd_large/common/VLMs/
export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/VLMs/datasets/

TASK_CONFIG=LandformRetrieval
MODEL_CONFIG=OpsMM
EXP_NAME=main_exp

QUERY_MODE=text

MODEL_NAME=OpenSearch-AI/Ops-MM-embedding-v1-2B
PRETRAINED=opsmm
IMAGE_ENCODER_TYPE=opsmm_v1
TEXT_ENCODER_TYPE=opsmm_v1

PROMPT_COUNTS=(1 3 5 7 10)

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
