#!/usr/bin/env bash
# Usage: bash scripts/vlm2vec.sh 0,1  (or export CUDA_VISIBLE_DEVICES beforehand)
export TORCH_DISTRIBUTED_TIMEOUT=86400
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH

# Huggingface mirrors/caches
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/mnt/sharedata/ssd_large/common/VLMs/}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/mnt/sharedata/ssd_large/common/VLMs/datasets/}

TASK_CONFIG=GlobalGeoLocalization
MODEL_CONFIG=VLM2Vec
EXP_NAME=vlm2vec_exp

# Query settings
QUERY_MODE=text   # image | text | hybrid
QUERY_TEXT=yardangs
QUERY_IMAGES=/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_geolocalization/image_queries/${QUERY_TEXT}
GROUND_TRUTH_CSV=/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_geolocalization/dataset/ground_truth/${QUERY_TEXT}.csv

# VLM2Vec model (Qwen2-VL backbone)
MODEL_NAME=VLM2Vec/VLM2Vec-V2.0
PRETRAINED=vlm2vec # 仅用于日志/命名
IMAGE_ENCODER_TYPE=vlm2vec
TEXT_ENCODER_TYPE=vlm2vec

# Optional distributed DB build (skips if DB already exists)
NPROC=$(( $(echo "${CUDA_VISIBLE_DEVICES:-}" | tr -cd ',' | wc -c) + 1 ))
if [[ "${NPROC}" -gt 1 ]]; then
  echo "Building DB with torchrun on ${NPROC} GPUs: ${CUDA_VISIBLE_DEVICES}"
  torchrun --nproc_per_node=${NPROC} build_db.py \
    --task_config "${TASK_CONFIG}" \
    --model_config "${MODEL_CONFIG}" \
    --exp_name "${EXP_NAME}" \
    --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
    --text_encoder_type "none" \
    --model_name "${MODEL_NAME}" \
    --pretrained "${PRETRAINED}"
fi

# Run retrieval
python main.py \
  --task_config "${TASK_CONFIG}" \
    --model_config "${MODEL_CONFIG}" \
  --exp_name "${EXP_NAME}" \
  --query_mode "${QUERY_MODE}" \
  --query_text "${QUERY_TEXT}" \
  --query_images ${QUERY_IMAGES} \
  --ground_truth_csv "${GROUND_TRUTH_CSV}" \
  --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
  --text_encoder_type "${TEXT_ENCODER_TYPE}" \
  --model_name "${MODEL_NAME}" \
  --pretrained "${PRETRAINED}"
