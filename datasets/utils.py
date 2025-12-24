from .mars import MarsDatabaseBuilder


def build_dataset(args, image_encoder, delta: float = 0.2):
    builder = MarsDatabaseBuilder()
    return builder.build(args, image_encoder, delta=delta)
