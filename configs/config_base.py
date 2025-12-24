import os
import importlib
from dataclasses import dataclass


@dataclass
class Config:
    # --------------------- General ---------------------
    name = "open-clip"
    logs = "./logs/"
    project_name = None
    precision = "amp"
    seed = 0

    # --------------------- Encoders/Model ---------------------
    image_encoder_type = "openclip"
    text_encoder_type = None
    model = "RN50"
    pretrained = ""
    force_quick_gelu = False
    torchscript = False
    force_custom_text = None
    force_patch_dropout = None
    force_image_size = None
    image_mean = None
    image_std = None
    image_interpolation = None
    image_resize_mode = None
    aug_cfg = None
    pretrained_image = False
    cache_dir = None
    siglip = False

    # --------------------- Data / DB / Eval ---------------------
    project_dir = None
    database_root = None
    delta_degree = 0.2
    workers = 8
    batch_size_database = 128
    top_k = 100000
    eval_max_k = 200000
    radius_deg = 0.5
    feature_dim = 768
    mix_image_ratio = 0.3

    def __post_init__(self):
        args = self
        args.name = self.__class__.__name__
        args.output_dir = os.path.join(args.logs, args.name)


def load_static_config(config_name, type: str = "retrieval"):
    project_dir = os.path.dirname(__file__)
    all_configs = {}
    for file_name in os.listdir(project_dir):
        if file_name.endswith(".py") and file_name.startswith(f"configs_{type}"):
            module_name = file_name[:-3]
            full_module_name = f"{__package__}.{module_name}"
            module = importlib.import_module(full_module_name)
            for attr_name in dir(module):
                if attr_name in ["Config"] or attr_name.startswith("__") or attr_name.startswith("configs_static"):
                    continue
                if attr_name not in all_configs:
                    all_configs[attr_name] = module

    if config_name not in all_configs:
        raise KeyError(f"Config {config_name} not found in {type} configs.")
    config = getattr(all_configs[config_name], config_name)()
    return config
