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
    args.project_name = args_dynamic.project_name or args.project_name
    args.name = args_dynamic.exp_name or f"{args_dynamic.query_mode}_retrieval"
    args.query_mode = args_dynamic.query_mode
    explicit_image_type = args_dynamic.image_encoder_type
    args.image_encoder_type = explicit_image_type or getattr(args, "image_encoder_type", "openclip")
    args.text_encoder_type = args_dynamic.text_encoder_type or getattr(args, "text_encoder_type", None)
    model_spec = args_dynamic.model_name or args.model
    model_family, model_name = _parse_model_spec(model_spec, args.image_encoder_type)
    args.model = model_name
    # If the model spec carried a family prefix, sync image encoder type unless explicitly overridden
    if explicit_image_type is None and model_family != args.image_encoder_type:
        args.image_encoder_type = model_family
    # Default text encoder type follows the (possibly inferred) image encoder type if not explicitly set
    if args.text_encoder_type is None:
        args.text_encoder_type = "openclip" if args.image_encoder_type == "openclip" else "none"
    if args_dynamic.pretrained:
        args.pretrained = args_dynamic.pretrained
    if args_dynamic.resume_post_train:
        args.resume_post_train = args_dynamic.resume_post_train
    if args_dynamic.top_k:
        args.top_k = args_dynamic.top_k
    if args_dynamic.db_dir:
        args.db_dir = args_dynamic.db_dir
    if not getattr(args, "db_dir", None):
        delta_val = getattr(args, "delta_degree", 0.2)
        if args.resume_post_train:
            parts = args.resume_post_train.strip("/").split("/")[-3:]
            if parts[-1].endswith(".pt"):
                parts[-1] = parts[-1].rsplit(".", 1)[0]
            tail = "_".join(parts)
            suffix = tail.replace("/", "_")
        else:
            suffix = args.pretrained or "pretrained"
        tag_parts = [args.image_encoder_type, args.model, suffix]
        tag = "_".join(str(p) for p in tag_parts)
        base_dir = getattr(args, "database_root", None) or getattr(args, "project_dir", ".")
        args.db_dir = f"{base_dir}/image_size_{args.force_image_size}_delta_{delta_val}/{tag}"
    args.delta_degree = getattr(args, "delta_degree", 0.2)
    # Output dir: structured by model/pretrain/resume/query
    if args_dynamic.output_dir:
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

        query_mode_tag = _slugify(args_dynamic.query_mode)
        query_images = args_dynamic.query_images or []
        query_text = args_dynamic.query_text or ""
        if query_text:
            query_tag = _slugify(query_text)[:80]
        elif query_images:
            query_tag = f"images_{len(query_images)}"
        else:
            query_tag = "query"

        args.output_dir = os.path.join(
            args.logs, exp_prefix, model_tag, pretrained_tag, resume_tag, query_mode_tag, query_tag
        )
    if args_dynamic.radius_deg is not None:
        args.radius_deg = args_dynamic.radius_deg
    if args_dynamic.eval_max_k is not None:
        args.eval_max_k = args_dynamic.eval_max_k
    if args_dynamic.ground_truth_csv is not None:
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
