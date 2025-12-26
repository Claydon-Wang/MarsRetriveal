import logging
from typing import Optional

from .image_encoder.base import ImageEncoderBase
from .image_encoder.bgevl_image_encoder import BGEVLImageEncoder
from .image_encoder.dinov3_image_encoder import DinoV3ImageEncoder
from .image_encoder.jina_image_encoder import JinaImageEncoder
from .image_encoder.e5v_image_encoder import E5VImageEncoder
from .image_encoder.openclip_image_encoder import OpenCLIPImageEncoder
from .bgevl import BGEVLComponents, build_bgevl_components
from .e5v import E5VComponents, build_e5v_components
from .jina import JinaComponents, build_jina_components
from .openclip import OpenCLIPComponents, build_openclip_components
from .text_encoder.base import TextEncoderBase
from .text_encoder.bgevl_text_encoder import BGEVLTextEncoder
from .text_encoder.e5v_text_encoder import E5VTextEncoder
from .text_encoder.jina_text_encoder import JinaTextEncoder
from .text_encoder.openclip_text_encoder import OpenCLIPTextEncoder


def _infer_image_encoder_type(args) -> str:
    if getattr(args, "image_encoder_type", None):
        return args.image_encoder_type
    model = getattr(args, "model", "")
    if "/" in model:
        family, _ = model.split("/", 1)
        family = family.lower()
        if family == "jinaai":
            return "jina"
        if "bge-vl" in model.lower():
            return "bge-vl"
        if "e5-v" in model.lower():
            return "e5-v"
        return family
    return "openclip"


def _get_openclip_components(args, device) -> OpenCLIPComponents:
    if getattr(args, "_openclip_components", None) is None:
        args._openclip_components = build_openclip_components(args, device)
        args.preprocess_train = args._openclip_components.preprocess_train
        args.preprocess_val = args._openclip_components.preprocess_val
    return args._openclip_components


def _get_jina_components(args, device) -> JinaComponents:
    if getattr(args, "_jina_components", None) is None:
        args._jina_components = build_jina_components(args, device)
    return args._jina_components


def _get_bgevl_components(args, device) -> BGEVLComponents:
    if getattr(args, "_bgevl_components", None) is None:
        args._bgevl_components = build_bgevl_components(args, device)
    return args._bgevl_components


def _get_e5v_components(args, device) -> E5VComponents:
    if getattr(args, "_e5v_components", None) is None:
        args._e5v_components = build_e5v_components(args, device)
    return args._e5v_components


def build_image_encoder(args, device) -> ImageEncoderBase:
    encoder_type = _infer_image_encoder_type(args)
    if encoder_type == "openclip":
        components = _get_openclip_components(args, device)
        return OpenCLIPImageEncoder(components, device)

    if encoder_type.lower() == "dinov3":
        model_id = getattr(args, "model", None) or "facebook/dinov3-vitl16-pretrain-lvd1689m"
        pooling = getattr(args, "dinov3_pooling", "cls")
        return DinoV3ImageEncoder(model_id, device, pooling=pooling)

    if encoder_type == "jina":
        components = _get_jina_components(args, device)
        return JinaImageEncoder(components)

    if encoder_type == "bge-vl":
        components = _get_bgevl_components(args, device)
        return BGEVLImageEncoder(components)

    if encoder_type == "e5-v":
        components = _get_e5v_components(args, device)
        return E5VImageEncoder(components)

    raise ValueError(f"Unsupported image encoder type: {encoder_type}")


def build_text_encoder(args, device) -> Optional[TextEncoderBase]:
    encoder_type = getattr(args, "text_encoder_type", None) or "openclip"
    if encoder_type == "openclip":
        components = _get_openclip_components(args, device)
        return OpenCLIPTextEncoder(components, device)

    if encoder_type == "jina":
        components = _get_jina_components(args, device)
        return JinaTextEncoder(components)

    if encoder_type == "bge-vl":
        components = _get_bgevl_components(args, device)
        return BGEVLTextEncoder(components)

    if encoder_type == "e5-v":
        components = _get_e5v_components(args, device)
        return E5VTextEncoder(components)

    if encoder_type == "none":
        logging.info("Text encoder disabled (type=none).")
        return None

    raise ValueError(f"Unsupported text encoder type: {encoder_type}")
