from .geolocalization_dataset import GeoLocalizationDatabaseBuilder


def build_dataset(args, image_encoder, delta: float = 0.2):
    task_name = getattr(args, "task_name", None) or "global_geolocalization"
    if task_name == "global_geolocalization":
        builder = GeoLocalizationDatabaseBuilder()
        return builder.build(args, image_encoder, delta=delta)
    raise ValueError(f"Unsupported task_name for dataset build: {task_name}")


def build_dataset_distributed(args, image_encoder, delta: float, rank: int, world_size: int):
    task_name = getattr(args, "task_name", None) or "global_geolocalization"
    if task_name == "global_geolocalization":
        builder = GeoLocalizationDatabaseBuilder()
        return builder.build_distributed(args, image_encoder, delta=delta, rank=rank, world_size=world_size)
    raise ValueError(f"Unsupported task_name for dataset build: {task_name}")
