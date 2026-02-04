#!/usr/bin/env bash
# Usage: bash scripts/landform_retrieval/marscope.sh 0,1,2,3 (or export CUDA_VISIBLE_DEVICES beforehand)
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH

# Huggingface settings
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/sharedata/ssd_large/common/VLMs/
export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/VLMs/datasets/

TASK_CONFIG=LandformRetrieval
MODEL_CONFIG=CLIPMarScope
EXP_NAME=main_exp

QUERY_MODE=text

MODEL_NAME=ViT-L-14-quickgelu
PRETRAINED=dfn2b
RESUME_POST_TRAINS=(
  "/mnt/sharedata/ssd_large/Planet/PlanetCLIP/model/logs/ckpt/ViT-L-14-quickgelu_dfn2b/checkpoints/epoch_10.pt"
  "/mnt/sharedata/ssd_large/Planet/PlanetCLIP/model/logs/post_training/ckpt/v2_mini_20251226_131303_ViT-L-14-quickgelu_dfn2b_None/checkpoints/epoch_latest.pt"
)
# /mnt/sharedata/ssd_large/Planet/PlanetCLIP/model/logs/ckpt/ViT-L-14-quickgelu_dfn2b/checkpoints/epoch_10.pt
# /mnt/sharedata/ssd_large/Planet/PlanetCLIP/model/logs/post_training/ckpt/v2_mini_20251226_131303_ViT-L-14-quickgelu_dfn2b_None/checkpoints/epoch_latest.pt
IMAGE_ENCODER_TYPE=openclip
TEXT_ENCODER_TYPE=openclip

PROMPT_COUNTS=(1 3 5 7 10)

for RESUME_POST_TRAIN in "${RESUME_POST_TRAINS[@]}"; do
  echo "Evaluating model with post-training checkpoint: ${RESUME_POST_TRAIN}"
  for PROMPT_COUNT in "${PROMPT_COUNTS[@]}"; do
    python main.py \
      --task_config "${TASK_CONFIG}" \
      --model_config "${MODEL_CONFIG}" \
      --exp_name "${EXP_NAME}_pc${PROMPT_COUNT}" \
      --query_mode "${QUERY_MODE}" \
      --model_name "${MODEL_NAME}" \
      --pretrained "${PRETRAINED}" \
      --resume_post_train "${RESUME_POST_TRAIN}" \
      --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
      --text_encoder_type "${TEXT_ENCODER_TYPE}" \
      --save_details \
      --prompt_count "${PROMPT_COUNT}"
  done
done
