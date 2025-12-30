import argparse
import json
import logging
import os
from types import SimpleNamespace

import pandas as pd
import torch
from PIL import Image

from configs.config_base import Config
from models.utils import build_image_encoder, build_text_encoder


SAMPLE_TEXTS = ["A dog in grass", "A cat in snow."]
SAMPLE_IMAGES = [
    "cache/American_Eskimo_Dog.jpg",
    "cache/Felis_catus-cat_on_snow.jpg",
]

# Per-model overrides. image_encoder_type/text_encoder_type are explicit to avoid inference issues.
MODEL_SPECS = {
    "e5-v": dict(
        label="e5-v",
        image_encoder_type="e5-v",
        text_encoder_type="e5-v",
        model="royokong/e5-v",
        patch_size=14,
    ),
    "jina": dict(
        label="jina-embeddings-v4",
        image_encoder_type="jina",
        text_encoder_type="jina",
        model="jinaai/jina-embeddings-v4",
    ),
    "b3_qwen2": dict(
        label="B3_Qwen2_2B",
        image_encoder_type="vlm2vec",
        text_encoder_type="vlm2vec",
        model="raghavlite/B3_Qwen2_2B",
    ),
    "vlm2vec_v2": dict(
        label="VLM2Vec-V2.0",
        image_encoder_type="vlm2vec",
        text_encoder_type="vlm2vec",
        model="VLM2Vec/VLM2Vec-V2.0",
    ),
    "opsmm_v1": dict(
        label="Ops-MM-embedding-v1-2B",
        image_encoder_type="opsmm_v1",
        text_encoder_type="opsmm_v1",
        model="OpenSearch-AI/Ops-MM-embedding-v1-2B",
        attn_implementation="flash_attention_2",
    ),
    "siglip2_512": dict(
        label="ViT-L-16-SigLIP2-512",
        image_encoder_type="openclip",
        text_encoder_type="openclip",
        model="ViT-L-16-SigLIP2-512",
        pretrained="hf-hub:timm/ViT-L-16-SigLIP2-512",
        siglip=True,
    ),
    "siglip_384": dict(
        label="ViT-L-16-SigLIP-384",
        image_encoder_type="openclip",
        text_encoder_type="openclip",
        model="ViT-L-16-SigLIP-384",
        pretrained="hf-hub:timm/ViT-L-16-SigLIP-384",
        siglip=True,
    ),
    "vitl14_dfn2b": dict(
        label="ViT-L-14-quickgelu-dfn2b",
        image_encoder_type="openclip",
        text_encoder_type="openclip",
        model="ViT-L-14-quickgelu",
        pretrained="dfn2b",
        force_quick_gelu=True,
    ),
    "pe_core": dict(
        label="PE-Core-L-14-336",
        image_encoder_type="openclip",
        text_encoder_type="openclip",
        model="PE-Core-L-14-336",
        pretrained="hf-hub:timm/PE-Core-L-14-336",
    ),
    "bge_vl_large": dict(
        label="BAAI/BGE-VL-large",
        image_encoder_type="bge-vl",
        text_encoder_type="bge-vl",
        model="BAAI/BGE-VL-large",
    ),

    "aimv2_large": dict(
        label="aimv2-large-patch14-224",
        image_encoder_type="aimv2_vl",
        text_encoder_type="aimv2_vl",
        model="apple/aimv2-large-patch14-224-lit",
        pretrained="c2cd59a786c4c06f39d199c50d08cc2eab9f8605",
    ),
    "gme": dict(
        label="gme-Qwen2-VL-2B-Instruct",
        image_encoder_type="gme",
        text_encoder_type="gme",
        model="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        gme_text_instruction="Find an image that matches the given text.",
    ),
}


def _load_images_for_encoder(image_encoder, image_paths):
    """Prepare images according to encoder expectations (path, PIL, or batched tensor)."""
    if getattr(image_encoder, "use_path_inputs", False):
        return image_paths

    preprocess = getattr(image_encoder, "get_processor", lambda: None)()
    pil_images = [Image.open(p).convert("RGB") for p in image_paths]
    if preprocess is None:
        return pil_images

    processed = [preprocess(img) for img in pil_images]
    # Stack torch tensors into a batch; otherwise return the processed list as-is.
    if processed and torch.is_tensor(processed[0]):
        return torch.stack(processed, dim=0)
    return processed


def _build_args(spec: dict):
    base = Config()  # provides sane defaults
    for k, v in spec.items():
        setattr(base, k, v)
    # Ensure required fields exist
    base.output_dir = getattr(base, "output_dir", "./logs/tmp_tests")
    base.workers = getattr(base, "workers", 0)
    return base


def run_one(spec_key: str, device: torch.device):
    spec = MODEL_SPECS[spec_key]
    args = _build_args(spec)
    logging.info("==== Testing %s ====", spec["label"])

    result = {
        "model_key": spec_key,
        "model_label": spec["label"],
        "status": "ok",
        "error": "",
        "image_feat_shape": "",
        "text_feat_shape": "",
        "similarity": "",
    }

    try:
        image_encoder = build_image_encoder(args, device)
        text_encoder = None
        if spec.get("text_encoder_type", "none") != "none":
            text_encoder = build_text_encoder(args, device)

        image_inputs = _load_images_for_encoder(image_encoder, SAMPLE_IMAGES)
        try:
            image_feats = image_encoder.encode_image(image_inputs)
        except Exception as exc_img:
            if spec_key == "e5-v":
                # e5-v is strict about image formats; retry with fresh PIL images.
                logging.warning("Retrying e5-v with raw PIL images due to: %s", exc_img)
                image_inputs = [Image.open(p).convert("RGB") for p in SAMPLE_IMAGES]
                image_feats = image_encoder.encode_image(image_inputs)
            else:
                raise
        result["image_feat_shape"] = str(tuple(image_feats.shape))

        if text_encoder is not None:
            text_feats = text_encoder.encode_text(SAMPLE_TEXTS)
            result["text_feat_shape"] = str(tuple(text_feats.shape))
            sims = (text_feats @ image_feats.T).detach().cpu().numpy().tolist()
            result["similarity"] = json.dumps(sims)
            logging.info("Similarity matrix (text x image): %s", sims)
        else:
            logging.info("Image encoder only; produced features shape: %s", tuple(image_feats.shape))
    except Exception as exc:  # pragma: no cover - diagnostic path
        logging.exception("Failed on model %s: %s", spec["label"], exc)
        result["status"] = "fail"
        result["error"] = str(exc)

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test multiple encoders.")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated keys to test (options: %s, or 'all')." % ",".join(MODEL_SPECS.keys()),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_csv", type=str, default="test.csv", help="Path to save results CSV.")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    device = torch.device(args.device)
    target_models = list(MODEL_SPECS.keys()) if args.models == "all" else args.models.split(",")
    out_path = args.output_csv
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    wrote_header = os.path.exists(out_path)

    for key in target_models:
        if key not in MODEL_SPECS:
            logging.warning("Unknown model key: %s (skipping)", key)
            continue
        res = run_one(key, device)
        df = pd.DataFrame([res])
        df.to_csv(out_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        logging.info("Appended result for %s to %s", key, out_path)


if __name__ == "__main__":
    main()
