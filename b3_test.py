import torch
from PIL import Image

from third_party.vlm2vec.src.arguments import ModelArguments, DataArguments
from third_party.vlm2vec.src.model.model import MMEBModel
from third_party.vlm2vec.src.model.processor import (
    load_processor,
    QWEN2_VL,
    VLM_IMAGE_TOKENS,
    Qwen2_VL_process_fn,
)
from third_party.vlm2vec.src.utils.basic_utils import batch_to_device

# ----------------------
# 1) init model + processor (B3 config)
# ----------------------
model_args = ModelArguments(
    model_name="raghavlite/B3_Qwen2_2B",  # 🔥 换成 B3
    pooling="last",                      # B3 / MMEB 默认
    normalize=True,                      # ⚠️ 检索一定要 normalize
    model_backbone="qwen2_vl",
    lora=True,                           # 🔥 B3 是 LoRA adapter
)
data_args = DataArguments()

processor = load_processor(model_args, data_args)
model = MMEBModel.load(model_args).to("cuda", dtype=torch.bfloat16)
model.eval()

# ----------------------
# 2) load images
# ----------------------
img_paths = [
    "/mnt/sharedata/ssd_large/users/wsy/project/planet/retrieval/MarsRetriveal/cache/American_Eskimo_Dog.jpg",
    "/mnt/sharedata/ssd_large/users/wsy/project/planet/retrieval/MarsRetriveal/cache/Felis_catus-cat_on_snow.jpg",
]
images = [Image.open(p).convert("RGB") for p in img_paths]

# ----------------------
# 3) compute image embeddings (image → qry_reps)
# ----------------------
qry_texts = [
    f"{VLM_IMAGE_TOKENS[QWEN2_VL]} Represent the given image with the following question: Describe this image"
    for _ in images
]

processor_inputs = {
    "text": qry_texts,
    "images": images,
}

qry_inputs = Qwen2_VL_process_fn(processor_inputs, processor)
qry_inputs = batch_to_device(qry_inputs, "cuda")

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    qry_reps = model(qry=qry_inputs)["qry_reps"]  # [B, d]

# ----------------------
# 4) compute text embeddings (text → tgt_reps)
# ----------------------
texts = [
    "A dog playing outside.",
    "A cat standing in the snow.",
]

processor_inputs = {
    "text": texts,
    "images": [None] * len(texts),
}

tgt_inputs = Qwen2_VL_process_fn(processor_inputs, processor)
tgt_inputs = batch_to_device(tgt_inputs, "cuda")

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    tgt_reps = model(tgt=tgt_inputs)["tgt_reps"]  # [T, d]

# ----------------------
# 5) similarity matrix
# ----------------------
sim_matrix = model.compute_similarity(qry_reps, tgt_reps)  # [B, T]
print(sim_matrix)
