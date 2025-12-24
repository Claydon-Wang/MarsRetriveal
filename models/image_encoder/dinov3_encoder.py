import logging

import torch
from torch.nn import functional as F

from .base import ImageEncoderBase


class DinoV3ImageEncoder(ImageEncoderBase):
    """Optional DinoV3 wrapper for image-only retrieval."""

    def __init__(self, checkpoint_path: str, device: torch.device, image_size: int = 448):
        try:
            import torchvision.transforms as T
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("torchvision is required for DINOv3 preprocessing.") from exc

        try:
            self.model = torch.hub.load(
                repo_or_dir=checkpoint_path,
                model="dinov3_vitb14",
                source="local",
                trust_repo=True,
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Failed to load DINOv3 model. Ensure checkpoint_path points to a local repo with hubconf.py."
            ) from exc

        self.model.eval().to(device)
        self.device = device
        self.preprocess = T.Compose(
            [
                T.Resize(image_size, antialias=True),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.model(images.to(self.device))
            feats = F.normalize(feats, p=2, dim=-1)
        return feats
