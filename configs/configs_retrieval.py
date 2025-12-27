from dataclasses import dataclass
from .config_base import Config

@dataclass
class MarsRetrieval(Config):
    project_dir = "/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_localization"
    database_root = "/mnt/sharedata/ssd_large/Planet/MarsRetrieval/global_localization/database"
    logs = "./logs"
    ground_truth_csv = None
    seed = 1
    top_k = 20000

    def __post_init__(self):
        super().__post_init__()
        self.global_img_dir = f"{self.project_dir}/dataset/mars_global.jpg" # Global mosaic image for plotting

@dataclass
class MarsRetrievalCLIP_MarScope(MarsRetrieval):
    model = "ViT-L-14-quickgelu" # ViT-L-14-quickgelu
    pretrained = "dfn2b"
    force_image_size = 512
    force_quick_gelu = True


@dataclass
class MarsRetrievalE5V(MarsRetrieval):
    image_encoder_type = "e5-v"
    text_encoder_type = "e5-v"
    model = "royokong/e5-v"
    batch_size_database = 1
    patch_size = 14


@dataclass
class MarsRetrievalVLM2Vec(MarsRetrieval):
    image_encoder_type = "vlm2vec"
    text_encoder_type = "vlm2vec"
    model = "VLM2Vec/VLM2Vec-V2.0"
    batch_size_database = 4



# @dataclass
# class MarsRetrievalCLIP_Pretrained(MarsRetrieval):
#     model = "ViT-B-16-quickgelu"
#     pretrained = "openai"


# @dataclass
# class MarsRetrievalDinoV3(MarsRetrieval):
#     model = "facebook/dinov3-vitl16-pretrain-lvd1689m"
