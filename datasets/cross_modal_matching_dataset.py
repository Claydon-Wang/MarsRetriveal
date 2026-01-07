import json
import logging
import os
import tarfile
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

from .base import DatasetBuilderBase

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CrossModalWDSDataset(IterableDataset):
    def __init__(self, tar_paths: List[str]):
        self.tar_paths = tar_paths

    def _iter_tar(self, tar_path: str):
        with tarfile.open(tar_path) as tar:
            pending = {}
            for member in tar:
                if not member.isfile():
                    continue
                name = member.name
                base, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext not in (".jpg", ".jpeg", ".png", ".json"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                if base not in pending:
                    pending[base] = {}
                if ext == ".json":
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        continue
                    final_caption = data.get("final_caption", {}) or {}
                    caption = final_caption.get("fused_caption") or data.get("caption")
                    pending[base]["caption"] = caption
                else:
                    try:
                        image = Image.open(f).convert("RGB")
                    except Exception:
                        continue
                    pending[base]["image"] = image
                if "image" in pending[base] and "caption" in pending[base]:
                    sample = pending.pop(base)
                    if sample["caption"]:
                        yield base, sample["image"], sample["caption"]

    def __iter__(self):
        worker_info = get_worker_info()
        tar_paths = self.tar_paths
        if worker_info is not None:
            per_worker = int(np.ceil(len(tar_paths) / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(tar_paths))
            tar_paths = tar_paths[start:end]
        for tar_path in tar_paths:
            yield from self._iter_tar(tar_path)


class CrossModalMatchingDatabaseBuilder(DatasetBuilderBase):
    def _resolve_paths(self, args, delta: float) -> Tuple[str, str]:
        if getattr(args, "db_dir", None):
            db_dir = args.db_dir
        else:
            base_dir = getattr(args, "database_root", None) or getattr(args, "project_dir", ".")
            suffix = args.pretrained or "pretrained"
            model_tag = str(args.model).replace("/", "_")
            tag = "_".join([model_tag, str(suffix)])
            db_dir = f"{base_dir}/{tag}"
        dataset_root = f"{args.project_dir}/dataset"
        return db_dir, dataset_root

    def build(self, args, image_encoder, text_encoder, delta: float) -> Dict:
        if text_encoder is None:
            raise ValueError("Text encoder is required for cross-modal matching.")

        db_dir, dataset_root = self._resolve_paths(args, delta)
        logging.info("Building cross-modal database at delta=%s -> %s", delta, db_dir)
        os.makedirs(db_dir, exist_ok=True)

        image_feat_path = os.path.join(db_dir, "image_features.npy")
        text_feat_path = os.path.join(db_dir, "text_features.npy")
        keys_path = os.path.join(db_dir, "keys.npy")

        if os.path.exists(image_feat_path) and os.path.exists(text_feat_path) and os.path.exists(keys_path):
            logging.info("Loading cached cross-modal database from %s", db_dir)
            image_features = np.load(image_feat_path)
            text_features = np.load(text_feat_path)
            keys = np.load(keys_path).tolist()
            if getattr(args, "feature_dim", None) is None:
                args.feature_dim = image_features.shape[1]
        else:
            tar_paths = sorted(
                os.path.join(dataset_root, fname)
                for fname in os.listdir(dataset_root)
                if fname.endswith(".tar")
            )
            dataset = CrossModalWDSDataset(tar_paths)
            encoder_collate = getattr(image_encoder, "collate_fn", None)
            preprocess = image_encoder.get_processor()
            total_pairs = 0
            for tar_path in tar_paths:
                try:
                    with tarfile.open(tar_path) as tar:
                        total_pairs += sum(1 for m in tar.getmembers() if m.name.endswith(".json"))
                except tarfile.TarError:
                    continue

            def _collate(batch):
                keys, images, captions = zip(*batch)
                if encoder_collate is not None:
                    enc_images, _ = encoder_collate(list(zip(images, keys)))
                else:
                    if preprocess is not None:
                        enc_images = torch.stack([preprocess(img) for img in images], dim=0)
                    else:
                        enc_images = list(images)
                return list(keys), enc_images, list(captions)

            loader = DataLoader(
                dataset,
                batch_size=args.batch_size_database,
                num_workers=args.workers,
                pin_memory=True,
                collate_fn=_collate,
            )

            image_feats = []
            text_feats = []
            keys = []

            total_batches = int(np.ceil(total_pairs / args.batch_size_database)) if total_pairs else None
            for batch_keys, batch_images, batch_caps in tqdm(
                loader,
                desc="Building cross-modal database",
                unit="batch",
                total=total_batches,
            ):
                with torch.no_grad():
                    img_features = image_encoder.encode_image(batch_images).cpu().numpy()
                    txt_features = text_encoder.encode_text(batch_caps).cpu().numpy()
                image_feats.append(img_features)
                text_feats.append(txt_features)
                keys.extend(batch_keys)

            image_features = np.concatenate(image_feats, axis=0).astype("float32")
            text_features = np.concatenate(text_feats, axis=0).astype("float32")
            if getattr(args, "feature_dim", None) is None:
                args.feature_dim = image_features.shape[1]

            np.save(image_feat_path, image_features)
            np.save(text_feat_path, text_features)
            np.save(keys_path, np.array(keys))
            logging.info("Saved cross-modal features to %s", db_dir)

        index_text = faiss.IndexFlatIP(text_features.shape[1])
        index_text.add(text_features)
        index_image = faiss.IndexFlatIP(image_features.shape[1])
        index_image.add(image_features)

        return {
            "image_features": image_features,
            "text_features": text_features,
            "keys": keys,
            "index_text": index_text,
            "index_image": index_image,
            "db_dir": db_dir,
            "dataset_root": dataset_root,
        }
