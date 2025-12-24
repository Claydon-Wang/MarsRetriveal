#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

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
DINOV3_CHECKPOINT=${DINOV3_CHECKPOINT:-"/path/to/dinov3_repo"}  # hubconf 目录
MODEL_NAME=${MODEL_NAME:-"dinov3_vitb14"}    # 仅用于日志/命名
PRETRAINED=${PRETRAINED:-"dinov3"}           # 仅用于日志/命名
RESUME_POST_TRAIN=${RESUME_POST_TRAIN:-""}

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
  --dinov3_checkpoint "${DINOV3_CHECKPOINT}" \
  --model_name "${MODEL_NAME}" \
  --pretrained "${PRETRAINED}" \
  ${RESUME_POST_TRAIN:+--resume_post_train "${RESUME_POST_TRAIN}"} \
  ${OUTPUT_DIR:+--output_dir "${OUTPUT_DIR}"}
