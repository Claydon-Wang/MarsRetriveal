from dataclasses import dataclass
from .config_base import Config

@dataclass
class MarsRetrieval(Config):
    project_dir = "/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_localization"
    database_root = "/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_localization/database"
    image_mean = (0.48145466, 0.4578275, 0.40821073)
    image_std = (0.26862954, 0.26130258, 0.27577711)
    model = "ViT-B-16-quickgelu"
    pretrained = "openai"
    force_image_size = 512
    force_quick_gelu = True
    resume_post_train = None
    logs = "./logs"
    ground_truth_csv = None
    feature_dim = 768
    seed = 1
    top_k = 20000

    def __post_init__(self):
        super().__post_init__()
        # Global mosaic image for plotting
        self.global_img_dir = f"{self.project_dir}/dataset/mars_global.jpg"


@dataclass
class MarsRetrievalSIGLIP(MarsRetrieval):
    image_mean = (0.5, 0.5, 0.5)
    image_std = (0.5, 0.5, 0.5)
    model = "ViT-L-16-SigLIP2-512"
    pretrained = "webli"
    force_image_size = 512
    feature_dim = 1024
    force_quick_gelu = False

@dataclass
class MarsRetrievalDinoV3(MarsRetrieval):
    feature_dim = 1024
