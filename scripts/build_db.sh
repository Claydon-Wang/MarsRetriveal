#!/usr/bin/env bash
# Distributed database builder (torchrun).

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Huggingface settings (override as needed)
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/mnt/sharedata/ssd_large/common/VLMs/}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/mnt/sharedata/ssd_large/common/VLMs/datasets/}

CONFIG_NAME=${CONFIG_NAME:-MarsRetrieval}
EXP_NAME=${EXP_NAME:-db_build}

# model / encoder
IMAGE_ENCODER_TYPE=${IMAGE_ENCODER_TYPE:-openclip}  # openclip | dinov3
TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-none}
MODEL_NAME=${MODEL_NAME:-ViT-B-16-quickgelu}        # e.g., ViT-B-16-quickgelu or facebook/dinov3-vitl16-pretrain-lvd1689m
PRETRAINED=${PRETRAINED:-openai}                   # tag for logging; not used for HF DINOv3
RESUME_POST_TRAIN=${RESUME_POST_TRAIN:-""}
DINOV3_POOLING=${DINOV3_POOLING:-cls}              # cls | mean (only for dinov3)

# DB build settings
DELTA_DEGREE=${DELTA_DEGREE:-0.2}
BATCH_SIZE_DATABASE=${BATCH_SIZE_DATABASE:-128}
NPROC=${NPROC:-4}

cmd=(torchrun --nproc_per_node=${NPROC} main_db_build.py
  --config_name "${CONFIG_NAME}"
  --exp_name "${EXP_NAME}"
  --image_encoder_type "${IMAGE_ENCODER_TYPE}"
  --text_encoder_type "${TEXT_ENCODER_TYPE}"
  --model_name "${MODEL_NAME}"
  --pretrained "${PRETRAINED}"
  --delta_degree "${DELTA_DEGREE}"
  --batch_size_database "${BATCH_SIZE_DATABASE}"
  ${DINOV3_POOLING:+--dinov3_pooling "${DINOV3_POOLING}"} )

[[ -n "${RESUME_POST_TRAIN}" ]] && cmd+=(--resume_post_train "${RESUME_POST_TRAIN}")
[[ -n "${OUTPUT_DIR:-}" ]] && cmd+=(--output_dir "${OUTPUT_DIR}")

echo "Running: ${cmd[*]}"
"${cmd[@]}"

