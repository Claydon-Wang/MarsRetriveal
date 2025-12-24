#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Huggingface settings (set your own token via env)
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_TOKEN=${HF_TOKEN:-""}
export HF_HOME=${HF_HOME:-/mnt/sharedata/ssd_large/common/VLMs/}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/mnt/sharedata/ssd_large/common/VLMs/datasets/}

CONFIG_NAME=MarsRetrieval
EXP_NAME=dinov3_exp

# query settings
QUERY_MODE=image   # image | text | hybrid (DINOv3 仅支持 image)
QUERY_TEXT=yardangs  # 用于 ground truth 选择，可用下划线代替空格
QUERY_IMAGES=${QUERY_IMAGES:-"/path/to/query.jpg"}  # 单图、目录或空格分隔多图
GROUND_TRUTH_CSV=${GROUND_TRUTH_CSV:-"/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_localization/dataset/ground_truth/${QUERY_TEXT}.csv"}

# model / encoder settings (DINOv3)
IMAGE_ENCODER_TYPE=dinov3
TEXT_ENCODER_TYPE=none
MODEL_NAME=facebook/dinov3-vitl16-pretrain-lvd1689m  # HF 模型 ID
PRETRAINED=hf  # 仅用于日志/命名

# run
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
  --pretrained "${PRETRAINED}" \
