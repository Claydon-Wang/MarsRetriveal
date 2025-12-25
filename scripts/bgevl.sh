export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH

# Hugging Face mirrors/caches (optional, adjust as needed)
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/mnt/sharedata/ssd_large/common/VLMs/}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/mnt/sharedata/ssd_large/common/VLMs/datasets/}

CONFIG_NAME=MarsRetrieval
EXP_NAME=main_exp

# Query settings
QUERY_MODE=text   # image | text | hybrid
QUERY_TEXT=yardangs
QUERY_IMAGES=/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_localization/image_queries/yardangs
GROUND_TRUTH_CSV=/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_localization/dataset/ground_truth/yardangs.csv

# BGE-VL model
MODEL_NAME=BAAI/BGE-VL-large   # or BAAI/BGE-VL-large
PRETRAINED=bge-vl #
IMAGE_ENCODER_TYPE=bge-vl
TEXT_ENCODER_TYPE=bge-vl

# optional distributed DB build (skips if DB already exists)
NPROC=$(( $(echo "${CUDA_VISIBLE_DEVICES:-}" | tr -cd ',' | wc -c) + 1 ))
if [[ "${NPROC}" -gt 1 ]]; then
  torchrun --nproc_per_node=${NPROC} build_db.py \
    --config_name "${CONFIG_NAME}" \
    --exp_name "${EXP_NAME}" \
    --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
    --text_encoder_type "none" \
    --model_name "${MODEL_NAME}" \
    --pretrained "${PRETRAINED}"
fi

# Run retrieval
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
