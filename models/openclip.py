import logging
from dataclasses import dataclass
from typing import Optional

import torch

try:
    from open_clip import create_model_and_transforms, get_tokenizer
    from open_clip.factory import load_checkpoint
except Exception as exc:  # pragma: no cover - imported dynamically
    logging.warning("Failed to import open_clip modules: %s", exc)
    create_model_and_transforms = None
    get_tokenizer = None
    load_checkpoint = None


@dataclass
class OpenCLIPComponents:
    model: object
    preprocess_train: object
    preprocess_val: object
    tokenizer: object


def build_openclip_components(args, device) -> OpenCLIPComponents:
    if create_model_and_transforms is None:
        raise ImportError("open_clip is not available in the current environment.")

    model_kwargs = {}
    if getattr(args, "siglip", False):
        model_kwargs["init_logit_scale"] = torch.log(torch.tensor(10.0))
        model_kwargs["init_logit_bias"] = -10

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        cache_dir=args.cache_dir,
        **model_kwargs,
    )

    if args.resume_post_train is not None and load_checkpoint is not None:
        logging.info("Loading checkpoint from %s", args.resume_post_train)
        incompatible_keys = load_checkpoint(
            model=model,
            checkpoint_path=args.resume_post_train,
            strict=False,
            weights_only=False,
            device="cpu",
        )
        if incompatible_keys and (incompatible_keys.missing_keys or incompatible_keys.unexpected_keys):
            logging.warning("Incompatible keys while loading checkpoint: %s", incompatible_keys)

    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir)
    return OpenCLIPComponents(
        model=model,
        preprocess_train=preprocess_train,
        preprocess_val=preprocess_val,
        tokenizer=tokenizer,
    )
