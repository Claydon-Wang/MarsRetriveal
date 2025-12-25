import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist

from configs.config_base import load_static_config
from tools.utils import _merge_args, _configure_logging, random_seed
from datasets.utils import build_dataset_distributed
from models.utils import build_image_encoder


def _parse_args():
    parser = argparse.ArgumentParser(description="Distributed database builder")
    parser.add_argument("--config_name", type=str, required=True, help="Static config name.")
    parser.add_argument("--project_name", type=str, default=None, help="Project name for logging.")
    parser.add_argument("--model_name", type=str, default=None, help="Model name override.")
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained tag override.")
    parser.add_argument("--resume_post_train", type=str, default=None, help="Checkpoint for pretrained weights.")
    parser.add_argument("--image_encoder_type", type=str, default=None, help="Image encoder type.")
    parser.add_argument("--text_encoder_type", type=str, default=None, help="Text encoder type.")
    parser.add_argument("--query_mode", type=str, default="image", help="Query mode placeholder (image/text/hybrid).")
    parser.add_argument("--query_images", nargs="*", default=None, help="Unused placeholder for parity.")
    parser.add_argument("--query_text", type=str, default=None, help="Unused placeholder for parity.")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name (for logs).")
    parser.add_argument("--output_dir", type=str, default=None, help="Unused here; kept for parity.")
    parser.add_argument("--delta_degree", type=float, default=None, help="Delta degree for DB build.")
    parser.add_argument("--dinov3_pooling", type=str, default=None, help="Pooling for DINOv3 (cls|mean).")
    parser.add_argument("--batch_size_database", type=int, default=None, help="Batch size for DB build.")
    parser.add_argument("--top_k", type=int, default=None, help="Unused placeholder for parity.")
    parser.add_argument("--db_dir", type=str, default=None, help="Optional database directory override.")
    parser.add_argument("--radius_deg", type=float, default=None, help="Radius in degrees (unused here).")
    parser.add_argument("--eval_max_k", type=int, default=None, help="Max K for eval (unused here).")
    parser.add_argument("--ground_truth_csv", type=str, default=None, help="GT CSV (unused here).")
    return parser.parse_args()


def init_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world_size, local_rank


def main():
    args_dyn = _parse_args()
    args = load_static_config(args_dyn.config_name, type="retrieval")
    args = _merge_args(args, args_dyn)

    rank, world_size, local_rank = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(device) if device.type == "cuda" else None

    if rank == 0:
        log_dir = os.path.join(args.logs, args.name)
        _configure_logging(log_dir)
        logging.info("World size: %s", world_size)
        logging.info("Using device: %s", device)
    dist.barrier()

    random_seed(args.seed + rank)

    image_encoder = build_image_encoder(args, device)
    delta = args_dyn.delta_degree if args_dyn.delta_degree is not None else getattr(args, "delta_degree", 0.2)

    result = build_dataset_distributed(args, image_encoder, delta=delta, rank=rank, world_size=world_size)
    if rank == 0:
        logging.info("DB build complete. Saved to %s", result.get("db_dir", ""))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
