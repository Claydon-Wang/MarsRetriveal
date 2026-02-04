#!/usr/bin/env bash
# Usage: bash scripts/openclip.sh 0,1,2,3 (or export CUDA_VISIBLE_DEVICES beforehand)
export CUDA_VISIBLE_DEVICES=$1
export PATH=~/.conda/envs/retrieval/bin:$PATH

# Huggingface settings
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/sharedata/ssd_large/common/VLMs/
export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/VLMs/datasets/

TASK_CONFIG=GlobalGeoLocalization
MODEL_CONFIG=CLIPMarScope
EXP_NAME=main_exp

# query settings
QUERY_MODES=(image text) # image / text / hybrid
QUERY_TEXTS=(alluvial_fans glacier-like_form landslides pitted_cones yardangs) # (alluvial_fans glacier-like_form landslides pitted_cones yardangs)


# model (OpenCLIP by default)
MODEL_NAME=ViT-L-14-quickgelu
PRETRAINED=dfn2b
RESUME_POST_TRAINS=(
  /mnt/sharedata/ssd_large/Planet/PlanetCLIP/model/logs/ckpt/ViT-L-14-quickgelu_dfn2b/checkpoints/epoch_10.pt
)
# /mnt/sharedata/ssd_large/Planet/PlanetCLIP/model/logs/ckpt/ViT-L-14-quickgelu_dfn2b/checkpoints/epoch_10.pt
# /mnt/sharedata/ssd_large/Planet/PlanetCLIP/model/logs/post_training/ckpt/v2_mini_20251226_131303_ViT-L-14-quickgelu_dfn2b_None/checkpoints/epoch_latest.pt

IMAGE_ENCODER_TYPE=openclip
TEXT_ENCODER_TYPE=openclip

for RESUME_POST_TRAIN in "${RESUME_POST_TRAINS[@]}"; do
  echo "Evaluating model with post-training checkpoint: ${RESUME_POST_TRAIN}"

  NPROC=$(( $(echo "${CUDA_VISIBLE_DEVICES:-}" | tr -cd ',' | wc -c) + 1 ))
  # optional distributed DB build (skips if DB already exists)
  if [[ "${NPROC}" -gt 1 ]]; then
    torchrun --nproc_per_node=${NPROC} --master_port=29501 build_db.py \
      --task_config "${TASK_CONFIG}" \
      --model_config "${MODEL_CONFIG}" \
      --exp_name "${EXP_NAME}" \
      --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
      --text_encoder_type "none" \
      --model_name "${MODEL_NAME}" \
      --pretrained "${PRETRAINED}" \
      --resume_post_train "${RESUME_POST_TRAIN}"
  fi

  # run retrieval for each landform
  for QUERY_MODE in "${QUERY_MODES[@]}"; do

    for QUERY_TEXT in "${QUERY_TEXTS[@]}"; do
      QUERY_IMAGES=/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_geolocalization/image_queries/${QUERY_TEXT}
      GROUND_TRUTH_CSV=/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_geolocalization/dataset/ground_truth/${QUERY_TEXT}.csv
      python main.py \
      --task_config "${TASK_CONFIG}" \
      --model_config "${MODEL_CONFIG}" \
      --exp_name "${EXP_NAME}" \
      --query_mode "${QUERY_MODE}" \
      --query_text "${QUERY_TEXT}" \
      --query_images ${QUERY_IMAGES} \
      --model_name "${MODEL_NAME}" \
      --pretrained "${PRETRAINED}" \
      --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
      --text_encoder_type "${TEXT_ENCODER_TYPE}" \
      --resume_post_train "${RESUME_POST_TRAIN}" \
      --ground_truth_csv "${GROUND_TRUTH_CSV}" \
      --save_details
    done

  done
  
done
