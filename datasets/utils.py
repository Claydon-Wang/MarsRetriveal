from .mars import MarsDatabaseBuilder


def build_dataset(args, image_encoder, delta: float = 0.2):
    builder = MarsDatabaseBuilder()
    return builder.build(args, image_encoder, delta=delta)


def build_dataset_distributed(args, image_encoder, delta: float, rank: int, world_size: int):
    builder = MarsDatabaseBuilder()
    return builder.build_distributed(args, image_encoder, delta=delta, rank=rank, world_size=world_size)
