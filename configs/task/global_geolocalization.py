from dataclasses import dataclass
from ..config_base import Config


@dataclass
class GlobalGeoLocalization(Config):
    task_name = "global_geolocalization"
    project_dir = "/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_geolocalization"
    database_root = "/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_geolocalization/database"
    logs = "./logs"
    ground_truth_csv = None
    seed = 1
    delta_degree = 0.2
    workers = 4
    batch_size_database = 256
    top_k = 20000
    eval_max_k = 20000
    radius_deg = 0.5
    feature_dim = None
    mix_image_ratio = 0.3

    def __post_init__(self):
        super().__post_init__()
        self.global_img_dir = f"{self.project_dir}/dataset/mars_global.jpg"
