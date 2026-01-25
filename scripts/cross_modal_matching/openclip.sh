#!/usr/bin/env bash
# Usage: bash scripts/cross_modal_matching/openclip.sh 0,1,2,3 (or export CUDA_VISIBLE_DEVICES beforehand)
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH

# Huggingface settings
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/sharedata/ssd_large/common/VLMs/
export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/VLMs/datasets/

TASK_CONFIG=CrossModalMatching
MODEL_CONFIG=OpenCLIP
EXP_NAME=main_exp

# MODEL_NAME=ViT-L-16-SigLIP2-512
# PRETRAINED=hf-hub:timm/ViT-L-16-SigLIP2-512
# MODEL_NAME=ViT-L-16-SigLIP-384
# PRETRAINED=hf-hub:timm/ViT-L-16-SigLIP-384
MODEL_NAME=ViT-L-14-quickgelu
PRETRAINED=dfn2b
IMAGE_ENCODER_TYPE=openclip
TEXT_ENCODER_TYPE=openclip

python main.py \
  --task_config "${TASK_CONFIG}" \
  --model_config "${MODEL_CONFIG}" \
  --exp_name "${EXP_NAME}" \
  --query_mode "text" \
  --model_name "${MODEL_NAME}" \
  --pretrained "${PRETRAINED}" \
  --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
  --text_encoder_type "${TEXT_ENCODER_TYPE}"
