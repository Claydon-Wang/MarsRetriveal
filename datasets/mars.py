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
import torch.distributed as dist
from numpy.lib.format import open_memmap

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
            db_dir = f"{base_dir}/delta_{delta}/{tag}"
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
            if getattr(args, "feature_dim", None) is None:
                args.feature_dim = features.shape[1]
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

            if getattr(args, "feature_dim", None) is None:
                args.feature_dim = features.shape[1]

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

        if features:
            features = np.concatenate(features, axis=0).astype("float32")
            if getattr(args, "feature_dim", None) is None:
                args.feature_dim = features.shape[1]
        else:
            dim = getattr(args, "feature_dim", None) or 0
            features = np.empty((0, dim), dtype="float32")

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
        db_dir, thumb_dir = self._resolve_paths(args, delta)
        shard_dir = os.path.join(db_dir, "shards")
        os.makedirs(shard_dir, exist_ok=True)

        feature_save_path = os.path.join(db_dir, "features.npy")
        metadata_save_path = os.path.join(db_dir, "metadata.pkl")
        coordinates_save_path = os.path.join(db_dir, "coordinates.pkl")

        # --- 0) 全rank一致判断 DB 是否存在（避免 collective 顺序不一致） ---
        local_exists = int(
            os.path.exists(feature_save_path)
            and os.path.exists(metadata_save_path)
            and os.path.exists(coordinates_save_path)
        )
        exists = torch.tensor(local_exists, device="cuda" if torch.cuda.is_available() else "cpu")
        dist.broadcast(exists, src=0)
        exists = int(exists.item())

        if exists:
            out = {}
            if rank == 0:
                features = np.load(feature_save_path, mmap_mode="r")
                with open(metadata_save_path, "rb") as f:
                    metadata = pickle.load(f)
                with open(coordinates_save_path, "rb") as f:
                    coordinates = pickle.load(f)
                if getattr(args, "feature_dim", None) is None:
                    args.feature_dim = features.shape[1]
                index = faiss.IndexFlatIP(args.feature_dim)
                # 分块 add，避免一次性复制
                bs = 200000
                for i in range(0, features.shape[0], bs):
                    index.add(np.asarray(features[i:i+bs]))
                out = {"index": index, "metadata": metadata, "coordinates": coordinates,
                    "db_dir": db_dir, "thumb_dir": thumb_dir}

            dist.barrier()  # 所有rank对齐一次后一起 return
            return out

        # --- 1) 每个rank构建自己的 shard ---
        dataset = MarsBenchmarkDataset(thumb_dir, transform=image_encoder.get_processor())
        indices = list(range(rank, len(dataset), world_size))
        self._build_shard(args, image_encoder, thumb_dir, indices, shard_dir, rank)

        print(f"[rank {rank}] before barrier shard", flush=True)
        dist.barrier()  # shard 完成（所有rank）
        print(f"[rank {rank}] after barrier shard", flush=True)

        # --- 2) rank0 合并 ---
        out = {}
        if rank == 0:
            feature_paths = [os.path.join(shard_dir, f"features_rank{r}.npy") for r in range(world_size)]
            metadata_paths = [os.path.join(shard_dir, f"metadata_rank{r}.pkl") for r in range(world_size)]
            coord_paths = [os.path.join(shard_dir, f"coordinates_rank{r}.pkl") for r in range(world_size)]

            shard_ns = []
            shard_dim = getattr(args, "feature_dim", None)
            for p in feature_paths:
                if os.path.exists(p):
                    arr = np.load(p, mmap_mode="r")
                    shard_ns.append(arr.shape[0])
                    if shard_dim is None and arr.shape[0] > 0:
                        shard_dim = arr.shape[1]
                else:
                    shard_ns.append(0)

            total_rows = sum(shard_ns)
            if shard_dim is None:
                shard_dim = args.feature_dim
            args.feature_dim = shard_dim
            D = shard_dim

            final = open_memmap(feature_save_path, mode="w+", dtype="float32", shape=(total_rows, D))
            off = 0
            for p, n in zip(feature_paths, shard_ns):
                if n == 0:
                    continue
                part = np.load(p, mmap_mode="r")
                final[off:off+n] = part
                off += n
            del final

            metadata, coordinates = [], []
            for mp, cp in zip(metadata_paths, coord_paths):
                if os.path.exists(mp):
                    with open(mp, "rb") as f:
                        metadata.extend(pickle.load(f))
                if os.path.exists(cp):
                    with open(cp, "rb") as f:
                        coordinates.extend(pickle.load(f))

            with open(metadata_save_path, "wb") as f:
                pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(coordinates_save_path, "wb") as f:
                pickle.dump(coordinates, f, protocol=pickle.HIGHEST_PROTOCOL)

            features = np.load(feature_save_path, mmap_mode="r")
            index = faiss.IndexFlatIP(D)
            bs = 200000
            for i in range(0, total_rows, bs):
                index.add(np.asarray(features[i:i+bs]))

            # clean shards
            for p in feature_paths + metadata_paths + coord_paths:
                if os.path.exists(p):
                    os.remove(p)
            try:
                os.rmdir(shard_dir)
            except OSError:
                pass

            out = {"index": index, "metadata": metadata, "coordinates": coordinates,
                "db_dir": db_dir, "thumb_dir": thumb_dir}

        print(f"[rank {rank}] before barrier merge", flush=True)
        dist.barrier()  # merge 完成（所有rank对齐）
        print(f"[rank {rank}] after barrier merge", flush=True)
        return out
