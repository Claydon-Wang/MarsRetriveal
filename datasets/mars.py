import logging
import os
import pickle
import re
from typing import Dict, Tuple, List

import faiss
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from .base import DatasetBuilderBase

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _extract_coordinates(image_name: str) -> Tuple[float, float]:
    match = re.match(r"^(E_|E)(\d+_\d+)_(N|N_)(\d+_\d+)\.png$", image_name)
    if match:
        lon_dir, lon_val_str, lat_dir, lat_val_str = match.groups()
        lon_sign = -1 if lon_dir == "E_" else 1
        lat_sign = 1 if lat_dir == "N" else -1
        lon_val = float(lon_val_str.replace("_", "."))
        lat_val = float(lat_val_str.replace("_", "."))
        return lon_sign * lon_val, lat_sign * lat_val
    return None, None


class MarsBenchmarkDataset(Dataset):
    def __init__(self, thumb_dir: str, transform=None):
        self.thumb_dir = thumb_dir
        self.transform = transform
        self.samples_pkl_path = os.path.join(os.path.dirname(thumb_dir), "samples.pkl")
        self.samples = self._load_or_scan_samples()

    def _load_or_scan_samples(self):
        if os.path.exists(self.samples_pkl_path):
            logging.info("Loading samples from %s", self.samples_pkl_path)
            with open(self.samples_pkl_path, "rb") as f:
                return pickle.load(f)

        logging.info("No samples.pkl found, scanning %s ...", self.thumb_dir)
        samples = []
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        all_files = [
            fname
            for fname in sorted(os.listdir(self.thumb_dir))
            if fname.lower().endswith(valid_extensions)
        ]
        for fname in tqdm(all_files, desc="Scanning images", unit="img"):
            samples.append(fname)

        with open(self.samples_pkl_path, "wb") as f:
            pickle.dump(samples, f)
        logging.info("Saved %d samples to %s", len(samples), self.samples_pkl_path)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_name = self.samples[index]
        path = os.path.join(self.thumb_dir, image_name)
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_name


class MarsDatabaseBuilder(DatasetBuilderBase):
    def _resolve_paths(self, args, delta: float) -> Tuple[str, str]:
        if getattr(args, "db_dir", None):
            db_dir = args.db_dir
        elif getattr(args, "resume_post_train", None):
            tag = args.resume_post_train.strip("/").split("/")[-3]
            base_dir = getattr(args, "database_root", None) or getattr(args, "project_dir", ".")
            db_dir = f"{base_dir}/image_size_{args.force_image_size}_delta_{delta}/{tag}"
        else:
            raise ValueError("db_dir or resume_post_train must be provided to locate the database.")

        thumb_dir = f"{args.project_dir}/dataset/global_images/thumb"
        return db_dir, thumb_dir

    def build(self, args, image_encoder, delta: float) -> Dict:
        db_dir, thumb_dir = self._resolve_paths(args, delta)
        logging.info("Building database at delta=%s -> %s", delta, db_dir)

        os.makedirs(db_dir, exist_ok=True)

        feature_save_path = os.path.join(db_dir, "features.npy")
        metadata_save_path = os.path.join(db_dir, "metadata.pkl")
        coordinates_save_path = os.path.join(db_dir, "coordinates.pkl")

        if (
            os.path.exists(feature_save_path)
            and os.path.exists(metadata_save_path)
            and os.path.exists(coordinates_save_path)
        ):
            logging.info("Loading cached database from %s", db_dir)
            features = np.load(feature_save_path)
            with open(metadata_save_path, "rb") as f:
                metadata = pickle.load(f)
            with open(coordinates_save_path, "rb") as f:
                coordinates = pickle.load(f)
        else:
            logging.info("Preparing database features from thumbnails in %s", thumb_dir)
            target_dataset = MarsBenchmarkDataset(thumb_dir, transform=image_encoder.get_processor())
            target_loader = DataLoader(
                target_dataset,
                batch_size=args.batch_size_database,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
            )

            features = []
            metadata = []
            coordinates = []

            for images, image_names in tqdm(target_loader, desc="Building database", unit="batch"):
                image_features = image_encoder.encode_image(images)
                image_features = image_features.cpu().numpy()

                features.append(image_features)
                metadata.extend(image_names)
                batch_coordinates = [_extract_coordinates(name) for name in image_names]
                coordinates.extend(batch_coordinates)

            features = np.concatenate(features, axis=0).astype("float32")

            np.save(feature_save_path, features)
            with open(metadata_save_path, "wb") as f:
                pickle.dump(metadata, f)
            with open(coordinates_save_path, "wb") as f:
                pickle.dump(coordinates, f)
            logging.info(
                "Saved features to %s, metadata to %s, coordinates to %s",
                feature_save_path,
                metadata_save_path,
                coordinates_save_path,
            )

        if features.shape[1] != args.feature_dim:
            raise ValueError(f"Feature dim mismatch: {features.shape[1]} != {args.feature_dim}")

        index = faiss.IndexFlatIP(args.feature_dim)
        index.add(features)

        return {
            "index": index,
            "metadata": metadata,
            "coordinates": coordinates,
            "db_dir": db_dir,
            "thumb_dir": thumb_dir,
        }

    def _build_shard(
        self, args, image_encoder, thumb_dir: str, indices: List[int], shard_dir: str, rank: int
    ):
        target_dataset = MarsBenchmarkDataset(thumb_dir, transform=image_encoder.get_processor())
        subset = Subset(target_dataset, indices)
        target_loader = DataLoader(
            subset,
            batch_size=args.batch_size_database,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )

        features = []
        metadata = []
        coordinates = []

        for images, image_names in tqdm(target_loader, desc=f"Rank {rank} building shard", unit="batch"):
            image_features = image_encoder.encode_image(images)
            image_features = image_features.cpu().numpy()

            features.append(image_features)
            metadata.extend(image_names)
            batch_coordinates = [_extract_coordinates(name) for name in image_names]
            coordinates.extend(batch_coordinates)

        features = np.concatenate(features, axis=0).astype("float32") if features else np.empty((0, args.feature_dim), dtype="float32")

        shard_paths = {
            "features": os.path.join(shard_dir, f"features_rank{rank}.npy"),
            "metadata": os.path.join(shard_dir, f"metadata_rank{rank}.pkl"),
            "coordinates": os.path.join(shard_dir, f"coordinates_rank{rank}.pkl"),
        }

        np.save(shard_paths["features"], features)
        with open(shard_paths["metadata"], "wb") as f:
            pickle.dump(metadata, f)
        with open(shard_paths["coordinates"], "wb") as f:
            pickle.dump(coordinates, f)
        return shard_paths

    def build_distributed(self, args, image_encoder, delta: float, rank: int, world_size: int) -> Dict:
        import torch.distributed as dist  # local import to avoid dependency when not used

        db_dir, thumb_dir = self._resolve_paths(args, delta)
        logging.info("Rank %s: building database at delta=%s -> %s", rank, delta, db_dir)

        shard_dir = os.path.join(db_dir, "shards")
        os.makedirs(shard_dir, exist_ok=True)

        feature_save_path = os.path.join(db_dir, "features.npy")
        metadata_save_path = os.path.join(db_dir, "metadata.pkl")
        coordinates_save_path = os.path.join(db_dir, "coordinates.pkl")

        # If DB already exists, rank 0 loads and others skip
        if rank == 0 and os.path.exists(feature_save_path) and os.path.exists(metadata_save_path) and os.path.exists(coordinates_save_path):
            logging.info("Rank 0: found existing database at %s, loading.", db_dir)
            features = np.load(feature_save_path)
            with open(metadata_save_path, "rb") as f:
                metadata = pickle.load(f)
            with open(coordinates_save_path, "rb") as f:
                coordinates = pickle.load(f)
            index = faiss.IndexFlatIP(args.feature_dim)
            index.add(features)
            dist.barrier()
            return {
                "index": index,
                "metadata": metadata,
                "coordinates": coordinates,
                "db_dir": db_dir,
                "thumb_dir": thumb_dir,
            }

        # Prepare indices for this rank
        dataset = MarsBenchmarkDataset(thumb_dir, transform=image_encoder.get_processor())
        indices = list(range(rank, len(dataset), world_size))

        shard_paths = self._build_shard(args, image_encoder, thumb_dir, indices, shard_dir, rank)

        dist.barrier()

        if rank == 0:
            feature_paths = []
            metadata_paths = []
            coord_paths = []
            for r in range(world_size):
                feature_paths.append(os.path.join(shard_dir, f"features_rank{r}.npy"))
                metadata_paths.append(os.path.join(shard_dir, f"metadata_rank{r}.pkl"))
                coord_paths.append(os.path.join(shard_dir, f"coordinates_rank{r}.pkl"))

            # Determine total rows without loading all features
            shard_sizes = []
            for p in feature_paths:
                if os.path.exists(p):
                    with np.load(p, mmap_mode="r") as arr:
                        shard_sizes.append(arr.shape[0])
                else:
                    shard_sizes.append(0)
            total_rows = sum(shard_sizes)

            # Memmap merge to avoid peak memory
            features_mm = np.memmap(feature_save_path, dtype="float32", mode="w+", shape=(total_rows, args.feature_dim))
            offset = 0
            for p, sz in zip(feature_paths, shard_sizes):
                if sz == 0 or not os.path.exists(p):
                    continue
                arr = np.load(p, mmap_mode="r")
                features_mm[offset : offset + sz] = arr
                offset += sz
            features = np.memmap(feature_save_path, dtype="float32", mode="r", shape=(total_rows, args.feature_dim))

            metadata = []
            coordinates = []
            for mp, cp in zip(metadata_paths, coord_paths):
                if os.path.exists(mp) and os.path.exists(cp):
                    with open(mp, "rb") as f:
                        metadata.extend(pickle.load(f))
                    with open(cp, "rb") as f:
                        coordinates.extend(pickle.load(f))

            os.makedirs(os.path.dirname(feature_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(metadata_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(coordinates_save_path), exist_ok=True)

            with open(metadata_save_path, "wb") as f:
                pickle.dump(metadata, f)
            with open(coordinates_save_path, "wb") as f:
                pickle.dump(coordinates, f)

            if features.shape[1] != args.feature_dim:
                raise ValueError(f"Feature dim mismatch: {features.shape[1]} != {args.feature_dim}")

            index = faiss.IndexFlatIP(args.feature_dim)
            index.add(np.array(features))

            logging.info("Rank 0: merged shards and built index with %d vectors.", features.shape[0])

            # clean shards after successful merge
            for p in feature_paths + metadata_paths + coord_paths:
                if os.path.exists(p):
                    os.remove(p)
            try:
                os.rmdir(shard_dir)
            except OSError:
                pass

            return {
                "index": index,
                "metadata": metadata,
                "coordinates": coordinates,
                "db_dir": db_dir,
                "thumb_dir": thumb_dir,
            }

        # Non-zero ranks return empty placeholder
        return {}
