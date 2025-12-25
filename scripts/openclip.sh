export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH

# Huggingface settings
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/sharedata/ssd_large/common/VLMs/
export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/VLMs/datasets/

CONFIG_NAME=MarsRetrieval
EXP_NAME=main_exp

# query settings
QUERY_MODE=image   # image / text / hybrid
QUERY_TEXT=yardangs # alluvial_fans / glacier-like_form / landslides / pitted_cones  / slope_streaks / yardangs  
QUERY_IMAGES=/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_localization/image_queries/${QUERY_TEXT}
GROUND_TRUTH_CSV=/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_localization/dataset/ground_truth/${QUERY_TEXT}.csv

# model (OpenCLIP by default)
# MODEL_NAME=ViT-L-14-quickgelu
# PRETRAINED=dfn2b
# MODEL_NAME=PE-Core-L-14-336
# PRETRAINED=hf-hub:timm/PE-Core-L-14-336
MODEL_NAME=ViT-L-16-SigLIP2-512
PRETRAINED=webli

IMAGE_ENCODER_TYPE=openclip
TEXT_ENCODER_TYPE=openclip


NPROC=$(( $(echo "${CUDA_VISIBLE_DEVICES:-}" | tr -cd ',' | wc -c) + 1 ))
# optional distributed DB build (skips if DB already exists)
if [[ "${NPROC}" -gt 1 ]]; then
  torchrun --nproc_per_node=${NPROC} build_db.py \
    --config_name "${CONFIG_NAME}" \
    --exp_name "${EXP_NAME}" \
    --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
    --text_encoder_type "none" \
    --model_name "${MODEL_NAME}" \
    --pretrained "${PRETRAINED}"
fi

# run retrieval
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
  --ground_truth_csv "${GROUND_TRUTH_CSV}"
