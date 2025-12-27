import torch
from PIL import Image
import numpy as np

# 官方提供的轻量 SDK
from third_party.ops_mm_embedding.ops_mm_embedding_v1 import OpsMMEmbeddingV1

# ----------------------
# 1) init model
# ----------------------
model = OpsMMEmbeddingV1(
    model_name="OpenSearch-AI/Ops-MM-embedding-v1-2B",
    device="cuda",                      # or "cpu"
    attn_implementation="flash_attention_2",  # 可选，加速
)

# ----------------------
# 2) load images
# ----------------------
img_paths = [
    "/mnt/sharedata/ssd_large/users/wsy/project/planet/retrieval/MarsRetriveal/cache/American_Eskimo_Dog.jpg",
    "/mnt/sharedata/ssd_large/users/wsy/project/planet/retrieval/MarsRetriveal/cache/Felis_catus-cat_on_snow.jpg",
]

images = [Image.open(p).convert("RGB") for p in img_paths]

# ----------------------
# 3) compute image embeddings
# ----------------------
with torch.no_grad():
    img_embeds = model.get_image_embeddings(images)   # [B, d]

# ⚠️ 官方实现已经做了 normalize，如需保险可再做一次
img_embeds = torch.nn.functional.normalize(img_embeds, dim=-1)

# ----------------------
# 4) compute text embeddings
# ----------------------
texts = [
    "A dog playing outside.",
    "A cat standing in the snow.",
]

with torch.no_grad():
    text_embeds = model.get_text_embeddings(texts)    # [T, d]

text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)

# ----------------------
# 5) similarity matrix
# ----------------------
# cosine similarity = dot product (after normalize)
sim_matrix = img_embeds @ text_embeds.T   # [B, T]

print(sim_matrix)
