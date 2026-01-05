import logging
import os
import pickle
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .base import DatasetBuilderBase

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LandformImageDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str, str]], transform=None, return_path: bool = False):
        self.samples = samples
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, image_name, label = self.samples[index]
        if self.return_path:
            return path, image_name, label
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_name, label


class LandformRetrievalDatabaseBuilder(DatasetBuilderBase):
    def _resolve_paths(self, args, delta: float) -> Tuple[str, str]:
        if getattr(args, "db_dir", None):
            db_dir = args.db_dir
        else:
            base_dir = getattr(args, "database_root", None) or getattr(args, "project_dir", ".")
            suffix = args.pretrained or "pretrained"
            model_tag = str(args.model).replace("/", "_")
            tag = "_".join([model_tag, str(suffix)])
            db_dir = f"{base_dir}/delta_{delta}/{tag}"
        dataset_root = f"{args.project_dir}/dataset"
        return db_dir, dataset_root

    def _scan_samples(self, dataset_root: str):
        class_dirs = [
            d for d in sorted(os.listdir(dataset_root))
            if os.path.isdir(os.path.join(dataset_root, d))
        ]
        samples = []
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        for cls in class_dirs:
            cls_dir = os.path.join(dataset_root, cls)
            for fname in sorted(os.listdir(cls_dir)):
                if os.path.splitext(fname)[1].lower() not in valid_exts:
                    continue
                full_path = os.path.join(cls_dir, fname)
                rel_name = os.path.relpath(full_path, dataset_root)
                samples.append((full_path, rel_name, cls))
        return samples, class_dirs

    def build(self, args, image_encoder, delta: float) -> Dict:
        db_dir, dataset_root = self._resolve_paths(args, delta)
        logging.info("Building landform database at delta=%s -> %s", delta, db_dir)
        os.makedirs(db_dir, exist_ok=True)

        feature_save_path = os.path.join(db_dir, "features.npy")
        metadata_save_path = os.path.join(db_dir, "metadata.pkl")
        labels_save_path = os.path.join(db_dir, "labels.pkl")
        classes_save_path = os.path.join(db_dir, "classes.pkl")

        if (
            os.path.exists(feature_save_path)
            and os.path.exists(metadata_save_path)
            and os.path.exists(labels_save_path)
            and os.path.exists(classes_save_path)
        ):
            logging.info("Loading cached database from %s", db_dir)
            features = np.load(feature_save_path)
            with open(metadata_save_path, "rb") as f:
                metadata = pickle.load(f)
            with open(labels_save_path, "rb") as f:
                labels = pickle.load(f)
            with open(classes_save_path, "rb") as f:
                classes = pickle.load(f)
            if getattr(args, "feature_dim", None) is None:
                args.feature_dim = features.shape[1]
        else:
            samples, classes = self._scan_samples(dataset_root)
            logging.info("Found %s classes and %s images", len(classes), len(samples))
            dataset = LandformImageDataset(
                samples,
                transform=image_encoder.get_processor(),
                return_path=getattr(image_encoder, "use_path_inputs", False),
            )
            encoder_collate = getattr(image_encoder, "collate_fn", None)

            def _collate(batch):
                images, image_names, batch_labels = zip(*batch)
                if encoder_collate is not None:
                    enc_images, enc_names = encoder_collate(list(zip(images, image_names)))
                else:
                    if torch.is_tensor(images[0]):
                        enc_images = torch.stack(images, dim=0)
                    else:
                        enc_images = list(images)
                    enc_names = list(image_names)
                return enc_images, enc_names, list(batch_labels)

            loader = DataLoader(
                dataset,
                batch_size=args.batch_size_database,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=_collate,
            )

            features = []
            metadata = []
            labels = []
            for batch in tqdm(loader, desc="Building landform database", unit="batch"):
                images, image_names, batch_labels = batch
                image_features = image_encoder.encode_image(images)
                image_features = image_features.cpu().numpy()
                features.append(image_features)
                metadata.extend(image_names)
                labels.extend(batch_labels)

            features = np.concatenate(features, axis=0).astype("float32")
            if getattr(args, "feature_dim", None) is None:
                args.feature_dim = features.shape[1]

            np.save(feature_save_path, features)
            with open(metadata_save_path, "wb") as f:
                pickle.dump(metadata, f)
            with open(labels_save_path, "wb") as f:
                pickle.dump(labels, f)
            with open(classes_save_path, "wb") as f:
                pickle.dump(classes, f)
            logging.info(
                "Saved features to %s, metadata to %s, labels to %s",
                feature_save_path,
                metadata_save_path,
                labels_save_path,
            )

        if features.shape[1] != args.feature_dim:
            raise ValueError(f"Feature dim mismatch: {features.shape[1]} != {args.feature_dim}")

        index = faiss.IndexFlatIP(args.feature_dim)
        index.add(features)

        gt_map = {}
        for img_name, label in zip(metadata, labels):
            gt_map.setdefault(label, set()).add(img_name)

        args.landform_gt = gt_map
        args.landform_classes = list(gt_map.keys())

        return {
            "index": index,
            "metadata": metadata,
            "labels": labels,
            "db_dir": db_dir,
            "dataset_root": dataset_root,
        }

    def build_distributed(self, args, image_encoder, delta: float, rank: int, world_size: int) -> Dict:
        raise NotImplementedError("Distributed DB build not implemented for landform retrieval.")
