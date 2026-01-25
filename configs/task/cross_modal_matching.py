from dataclasses import dataclass
from ..config_base import Config


@dataclass
class CrossModalMatching(Config):
    task_name = "cross_modal_matching"
    project_dir = "/mnt/sharedata/ssd_large/Planet/MarsRetrieval/cross_modal_matching/"
    database_root = "/mnt/sharedata/ssd_large/Planet/MarsRetrieval/cross_modal_matching/database/"
    caption_key = "qwen3_fused_caption" # ori_caption qwen3_fused_caption
    logs = "./logs"
    seed = 1
    workers = 4
    batch_size_database = 64
    top_k = 10
    feature_dim = None
