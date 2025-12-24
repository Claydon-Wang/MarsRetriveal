#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
export PATH=~/.conda/envs/retrieval/bin:$PATH

# Huggingface settings
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/sharedata/ssd_large/common/VLMs/
export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/VLMs/datasets/

CONFIG_NAME=MarsRetrieval
EXP_NAME=main_exp

# query settings
QUERY_MODE=text   # image / text / hybrid
QUERY_TEXT=pitted_cones # alluvial_fans / glacier-like_form / landslides / pitted_cones  / slope_streaks / yardangs  
QUERY_IMAGES=/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_localization/image_queries/${QUERY_TEXT}
GROUND_TRUTH_CSV=/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_localization/dataset/ground_truth/${QUERY_TEXT}.csv

# model (OpenCLIP by default)
MODEL_NAME=ViT-L-14-quickgelu
PRETRAINED=dfn2b
RESUME_POST_TRAIN=/mnt/sharedata/ssd_large/Planet/PlanetCLIP/model/logs/ckpt/ViT-L-14-quickgelu_dfn2b/checkpoints/epoch_10.pt
IMAGE_ENCODER_TYPE=openclip
TEXT_ENCODER_TYPE=openclip
export HF_TOKEN=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

# run
python main.py \
  --config_name "${CONFIG_NAME}" \
  --exp_name "${EXP_NAME}" \
  --query_mode "${QUERY_MODE}" \
  --query_text "${QUERY_TEXT}" \
  --query_images ${QUERY_IMAGES} \
  --model_name "${MODEL_NAME}" \
  --pretrained "${PRETRAINED}" \
  --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
  --text_encoder_type "${TEXT_ENCODER_TYPE}" \
  --resume_post_train "${RESUME_POST_TRAIN}" \
  --ground_truth_csv "${GROUND_TRUTH_CSV}"
