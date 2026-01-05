from dataclasses import dataclass
from ..config_base import Config


@dataclass
class LandformRetrieval(Config):
    task_name = "landform_retrieval"
    project_dir = "/mnt/sharedata/ssd_large/Planet/MarsRetrieval/landform_retrieval"
    database_root = "/mnt/sharedata/ssd_large/Planet/MarsRetrieval/landform_retrieval/database"
    logs = "./logs"
    seed = 1
    workers = 4
    batch_size_database = 256
    top_k = None
    eval_max_k = None
    feature_dim = None
