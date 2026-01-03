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
    def __init__(self, thumb_dir: str, transform=None, return_path: bool = False):
        self.thumb_dir = thumb_dir
        self.transform = transform
        self.return_path = return_path
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
        if self.return_path:
            return path, image_name
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
            target_dataset = MarsBenchmarkDataset(
                thumb_dir,
                transform=image_encoder.get_processor(),
                return_path=getattr(image_encoder, "use_path_inputs", False),
            )
            target_loader = DataLoader(
                target_dataset,
                batch_size=args.batch_size_database,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=getattr(image_encoder, "collate_fn", None),
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
        target_dataset = MarsBenchmarkDataset(
            thumb_dir,
            transform=image_encoder.get_processor(),
            return_path=getattr(image_encoder, "use_path_inputs", False),
        )
        subset = Subset(target_dataset, indices)
        target_loader = DataLoader(
            subset,
            batch_size=args.batch_size_database,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=getattr(image_encoder, "collate_fn", None),
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
            import time  # 引入 time 模块用于休眠
            db_dir, thumb_dir = self._resolve_paths(args, delta)
            shard_dir = os.path.join(db_dir, "shards")
            os.makedirs(shard_dir, exist_ok=True)

            feature_save_path = os.path.join(db_dir, "features.npy")
            metadata_save_path = os.path.join(db_dir, "metadata.pkl")
            coordinates_save_path = os.path.join(db_dir, "coordinates.pkl")

            # --- 0) 全rank一致判断 DB 是否存在 ---
            local_exists = int(
                os.path.exists(feature_save_path)
                and os.path.exists(metadata_save_path)
                and os.path.exists(coordinates_save_path)
            )

            if torch.cuda.is_available():
                dev = torch.device(f"cuda:{torch.cuda.current_device()}")
            else:
                dev = torch.device("cpu")

            exists = torch.tensor(local_exists, device=dev, dtype=torch.int32)
            dist.broadcast(exists, src=0)
            exists = int(exists.item())

            if exists:
                # ... (这部分保持不变，加载已有DB的代码) ...
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
                    bs = 200000
                    for i in range(0, features.shape[0], bs):
                        index.add(np.asarray(features[i:i+bs]))
                    out = {"index": index, "metadata": metadata, "coordinates": coordinates,
                        "db_dir": db_dir, "thumb_dir": thumb_dir}
                _barrier()
                return out

            # --- 1) 每个rank构建自己的 shard ---
            dataset = MarsBenchmarkDataset(thumb_dir, transform=image_encoder.get_processor())
            indices = list(range(rank, len(dataset), world_size))
            self._build_shard(args, image_encoder, thumb_dir, indices, shard_dir, rank)

            # shard 完成，写一个本地 flag，让 rank0 轮询所有 shard 是否结束，避免 barrier 超时
            shard_done = os.path.join(shard_dir, f"shard_done_rank{rank}.flag")
            with open(shard_done, "w") as f:
                f.write("done")
            logging.info(f"Rank {rank} shard done, waiting for merge.")

            # --- 2) rank0 合并 (关键修改区域) ---
            out = {}
            
            # 定义一个简单的标志文件路径，用于通知其他 Rank
            flag_file = os.path.join(shard_dir, "merge_done.flag")

            if rank == 0:
                try:
                    # 等待所有 shard 文件就绪，避免因为单个 rank 较慢导致 barrier 超时
                    all_done = [os.path.join(shard_dir, f"shard_done_rank{r}.flag") for r in range(world_size)]
                    while not all(os.path.exists(p) for p in all_done):
                        time.sleep(10)

                    logging.info("Rank 0 starting merge...")
                    feature_paths = [os.path.join(shard_dir, f"features_rank{r}.npy") for r in range(world_size)]
                    metadata_paths = [os.path.join(shard_dir, f"metadata_rank{r}.pkl") for r in range(world_size)]
                    coord_paths = [os.path.join(shard_dir, f"coordinates_rank{r}.pkl") for r in range(world_size)]

                    # 计算总行数
                    shard_ns = []
                    shard_dim = getattr(args, "feature_dim", None)
                    for p in feature_paths:
                        if os.path.exists(p):
                            # 只读取 shape，不加载数据，极快
                            arr_shape = np.load(p, mmap_mode="r").shape
                            shard_ns.append(arr_shape[0])
                            if shard_dim is None and arr_shape[0] > 0:
                                shard_dim = arr_shape[1]
                        else:
                            shard_ns.append(0)

                    total_rows = sum(shard_ns)
                    if shard_dim is None:
                        shard_dim = args.feature_dim
                    args.feature_dim = shard_dim
                    D = shard_dim

                    # 开始写入大文件
                    final = open_memmap(feature_save_path, mode="w+", dtype="float32", shape=(total_rows, D))
                    off = 0
                    for i, (p, n) in enumerate(zip(feature_paths, shard_ns)):
                        if n == 0:
                            continue
                        # 逐个 shard 读取并写入
                        part = np.load(p, mmap_mode="r")
                        final[off:off+n] = part
                        off += n
                        # ### 优化：每处理一个，手动释放内存引用
                        del part 
                    del final  # 刷新到磁盘

                    # 合并 Metadata
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

                    # 构建 Index
                    features = np.load(feature_save_path, mmap_mode="r")
                    index = faiss.IndexFlatIP(D)
                    bs = 200000
                    for i in range(0, total_rows, bs):
                        index.add(np.asarray(features[i:i+bs]))

                    # 清理临时文件
                    for p in feature_paths + metadata_paths + coord_paths:
                        if os.path.exists(p):
                            os.remove(p)
                    
                    out = {"index": index, "metadata": metadata, "coordinates": coordinates,
                        "db_dir": db_dir, "thumb_dir": thumb_dir}
                    
                    # ### 关键步骤：Rank 0 工作完成后，写入一个标志文件
                    with open(flag_file, "w") as f:
                        f.write("done")
                    logging.info("Rank 0 merge finished.")

                except Exception as e:
                    logging.error(f"Rank 0 merge failed: {e}")
                    # 即使失败也要抛出，防止其他进程死锁
                    raise e
            
            else:
                # ### 关键修改：Rank 1-7 不直接用 barrier，而是轮询检查文件
                # 这样它们会 sleep 释放 CPU 资源给 Rank 0
                logging.info(f"Rank {rank} waiting for Rank 0 to merge...")
                while not os.path.exists(flag_file):
                    time.sleep(2)  # 每 2 秒看一眼，不占 CPU
                logging.info(f"Rank {rank} detected merge done.")

            # 这里再用 barrier 确保大家状态一致，此时不会卡死，因为 Rank 0 已经完事了
            _barrier()
            
            # Rank 0 负责清理标志文件
            if rank == 0:
                try:
                    for r in range(world_size):
                        done_flag = os.path.join(shard_dir, f"shard_done_rank{r}.flag")
                        if os.path.exists(done_flag):
                            os.remove(done_flag)
                    os.remove(flag_file)
                    os.rmdir(shard_dir)
                except OSError:
                    pass

            return out

def _barrier():
    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()