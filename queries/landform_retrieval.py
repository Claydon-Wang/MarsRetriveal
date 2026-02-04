import logging
from typing import List

import numpy as np
import torch
from torch.nn import functional as F


PROMPT_TEMPLATES = [
    # --- N=1 (Baseline: 最通用的描述) ---
    "a photo of {}, a type of martian terrain",

    # --- N=3 (Standard: 增加遥感/卫星视角) ---
    "a satellite photo of {}.",
    "a high-resolution remote sensing image of {} on Mars.",

    # --- N=5 (Scientific: 增加地质学专业术语) ---
    "a detailed satellite view of {} geomorphic feature from Mars.",
    "an aerial perspective of {} morphological structure on Mars.",

    # --- N=7 (Domain Specific: 增加数据源/火星别称 - 关键涨点！) ---
    "high-definition remote sensing photo of {} on the red planet.",
    "a CTX or HiRISE image showing {} terrain on Mars.",  # <--- 最强的一个模板

    # --- N=10 (Full Ensemble: 最大化语义覆盖) ---
    "an orbital image of {} landform on the Martian surface.",
    "Martian {} as captured in space-based imagery.",
    "planetary-scale capture of {} in Martian geology."
]


def _format_concept(name: str) -> str:
    return name.replace("_", " ").replace("-", " ")


def build_landform_query(args, text_encoder, query_mode: str):
    if query_mode != "text":
        raise ValueError("Landform retrieval only supports text query_mode.")
    if text_encoder is None:
        raise ValueError("Text encoder is required for landform retrieval.")

    class_names: List[str] = getattr(args, "landform_classes", None) or []
    if not class_names:
        raise ValueError("No landform classes found; ensure dataset build ran before querying.")

    queries = []
    query_names = []
    with torch.no_grad():
        for cls in class_names:
            phrase = _format_concept(cls)
            prompt_count = getattr(args, "prompt_count", None)
            if prompt_count is None:
                prompt_count = len(PROMPT_TEMPLATES)
            prompts = [template.format(phrase) for template in PROMPT_TEMPLATES[:prompt_count]]
            if cls == class_names[0]:
                logging.info("Using %d prompt templates for landform retrieval.", len(prompts))
            feats = text_encoder.encode_text(prompts)
            if len(prompts) > 1:
                query = feats.mean(dim=0, keepdim=True)
                query = F.normalize(query, p=2, dim=-1)
            else:
                query = feats
            queries.append(query)
            query_names.append(cls)

    query_tensor = torch.cat(queries, dim=0)
    logging.info("Built %s landform text queries", len(query_names))
    return query_tensor.cpu().numpy(), query_names
