#!/usr/bin/env bash
# Usage: bash scripts/dinov3.sh 0,1,2,3  (or export CUDA_VISIBLE_DEVICES beforehand)
export TORCH_DISTRIBUTED_TIMEOUT=86400
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH

# Huggingface settings
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/sharedata/ssd_large/common/VLMs/
export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/VLMs/datasets/

TASK_CONFIG=GlobalGeoLocalization
MODEL_CONFIG=Jina
EXP_NAME=main_exp

# query settings
QUERY_MODE=image   # image | text | hybrid (DINOv3 仅支持 image)
QUERY_TEXT=yardangs  # 用于 ground truth 选择，可用下划线代替空格
QUERY_IMAGES=/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_geolocalization/image_queries/${QUERY_TEXT}
GROUND_TRUTH_CSV=/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_geolocalization/dataset/ground_truth/${QUERY_TEXT}.csv

# model / encoder settings (DINOv3)
IMAGE_ENCODER_TYPE=jina
TEXT_ENCODER_TYPE=jina
MODEL_NAME=jinaai/jina-embeddings-v4  # HF 模型 ID
PRETRAINED=hf  # 仅用于日志/命名

# optional distributed DB build (skips if DB already exists)
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

# run retrieval
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
  --pretrained "${PRETRAINED}" \
