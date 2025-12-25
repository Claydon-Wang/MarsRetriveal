import random
import numpy as np
import torch
import os
import logging
import sys
from datetime import datetime
from typing import List, Optional, Iterable


def random_seed(seed=42, rank=0):

    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)

        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False

def _slugify(val: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in val)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def _parse_model_spec(model: str, default_family: str = "openclip"):
    if "/" in model:
        family, name = model.split("/", 1)
        return family.lower(), name
    return default_family, model

def _merge_args(args, args_dynamic):
    args.project_name = getattr(args_dynamic, "project_name", None) or getattr(args, "project_name", None)
    query_mode_in = getattr(args_dynamic, "query_mode", None) or getattr(args, "query_mode", "image")
    args.name = getattr(args_dynamic, "exp_name", None) or f"{query_mode_in}_retrieval"
    args.query_mode = query_mode_in

    explicit_image_type = getattr(args_dynamic, "image_encoder_type", None)
    args.image_encoder_type = explicit_image_type or getattr(args, "image_encoder_type", "openclip")
    args.text_encoder_type = getattr(args_dynamic, "text_encoder_type", None) or getattr(args, "text_encoder_type", None)
    model_spec = getattr(args_dynamic, "model_name", None) or args.model
    # Non-openclip encoders: keep full model_spec (e.g., HF IDs with '/')
    if args.image_encoder_type != "openclip":
        model_family, model_name = args.image_encoder_type, model_spec
    else:
        if "/" in model_spec:
            model_family, model_name = _parse_model_spec(model_spec, args.image_encoder_type)
        else:
            model_family, model_name = args.image_encoder_type, model_spec
    args.model = model_name
    # If the model spec carried a family prefix, sync image encoder type unless explicitly overridden
    if explicit_image_type is None and model_family != args.image_encoder_type:
        args.image_encoder_type = model_family
    # Default text encoder type follows the (possibly inferred) image encoder type if not explicitly set
    if args.text_encoder_type is None:
        args.text_encoder_type = "openclip" if args.image_encoder_type == "openclip" else "none"
    if getattr(args_dynamic, "pretrained", None):
        args.pretrained = args_dynamic.pretrained
    if getattr(args_dynamic, "resume_post_train", None):
        args.resume_post_train = args_dynamic.resume_post_train
    if getattr(args_dynamic, "top_k", None):
        args.top_k = args_dynamic.top_k
    if getattr(args_dynamic, "db_dir", None):
        args.db_dir = args_dynamic.db_dir
    if not getattr(args, "db_dir", None):
        delta_val = getattr(args, "delta_degree", 0.2)
        if getattr(args_dynamic, "resume_post_train", None):
            parts = args.resume_post_train.strip("/").split("/")[-3:]
            if parts[-1].endswith(".pt"):
                parts[-1] = parts[-1].rsplit(".", 1)[0]
            tail = "_".join(parts)
            suffix = tail.replace("/", "_")
        else:
            suffix = args.pretrained or "pretrained"
        model_tag = str(args.model).replace("/", "_")
        tag = "_".join([model_tag, str(suffix)])
        base_dir = getattr(args, "database_root", None) or getattr(args, "project_dir", ".")
        args.db_dir = f"{base_dir}/delta_{delta_val}/{tag}"
    args.delta_degree = getattr(args, "delta_degree", 0.2)
    # Output dir: structured by model/pretrain/resume/query
    if getattr(args_dynamic, "output_dir", None):
        args.output_dir = args_dynamic.output_dir
    else:
        exp_prefix = _slugify(args.name)
        model_tag = _slugify(f"{args.image_encoder_type}_{args.model}")
        pretrained_tag = _slugify(args.pretrained or "pretrained")
        if args.resume_post_train:
            parts = args.resume_post_train.strip("/").split("/")[-3:]
            if parts and parts[-1].endswith(".pt"):
                parts[-1] = parts[-1].rsplit(".", 1)[0]
            resume_tag = _slugify("_".join(parts))
        else:
            resume_tag = "no_resume"

        query_mode_tag = _slugify(args.query_mode)
        query_images = getattr(args_dynamic, "query_images", None) or []
        query_text = getattr(args_dynamic, "query_text", None) or ""
        if query_text:
            query_tag = _slugify(query_text)[:80]
        elif query_images:
            query_tag = f"images_{len(query_images)}"
        else:
            query_tag = "query"

        args.output_dir = os.path.join(
            args.logs, exp_prefix, model_tag, pretrained_tag, resume_tag, query_mode_tag, query_tag
        )
    if getattr(args_dynamic, "radius_deg", None) is not None:
        args.radius_deg = args_dynamic.radius_deg
    if getattr(args_dynamic, "eval_max_k", None) is not None:
        args.eval_max_k = args_dynamic.eval_max_k
    if getattr(args_dynamic, "ground_truth_csv", None) is not None:
        args.ground_truth_csv = args_dynamic.ground_truth_csv
    return args



def _configure_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "retrieval.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="a")],
        force=True,
    )
    return log_file




def _validate_inputs(query_mode: str, images: List[str], query_text: str):
    if query_mode == "image" and not images:
        raise ValueError("Image mode requires --query_images.")
    if query_mode == "text" and not query_text:
        raise ValueError("Text mode requires --query_text.")
    if query_mode == "hybrid" and (not images or not query_text):
        raise ValueError("Hybrid mode requires both --query_images and --query_text.")
