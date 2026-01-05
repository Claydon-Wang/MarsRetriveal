#!/usr/bin/env bash
# Usage: bash scripts/landform_retrieval/marscope.sh 0,1,2,3 (or export CUDA_VISIBLE_DEVICES beforehand)
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH

# Huggingface settings
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/mnt/sharedata/ssd_large/common/VLMs/}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/mnt/sharedata/ssd_large/common/VLMs/datasets/}

TASK_CONFIG=LandformRetrieval
MODEL_CONFIG=CLIPMarScope
EXP_NAME=main_exp

QUERY_MODE=text

MODEL_NAME=ViT-L-14-quickgelu
PRETRAINED=dfn2b
RESUME_POST_TRAIN=/mnt/sharedata/ssd_large/Planet/PlanetCLIP/model/logs/ckpt/ViT-L-14-quickgelu_dfn2b/checkpoints/epoch_10.pt
IMAGE_ENCODER_TYPE=openclip
TEXT_ENCODER_TYPE=openclip

python main.py \
  --task_config "${TASK_CONFIG}" \
  --model_config "${MODEL_CONFIG}" \
  --exp_name "${EXP_NAME}" \
  --query_mode "${QUERY_MODE}" \
  --model_name "${MODEL_NAME}" \
  --pretrained "${PRETRAINED}" \
  --resume_post_train "${RESUME_POST_TRAIN}" \
  --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
  --text_encoder_type "${TEXT_ENCODER_TYPE}"
