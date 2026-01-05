from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import RetrieverBase


@dataclass
class LandformRetriever(RetrieverBase):
    args: object
    database: Dict

    def search(self, query, query_names: Optional[List[str]] = None) -> Dict:
        index = self.database["index"]
        metadata = self.database["metadata"]
        labels = self.database["labels"]

        if getattr(self.args, "top_k", None) is None or self.args.top_k <= 0:
            top_k = index.ntotal
        else:
            top_k = min(self.args.top_k, index.ntotal)

        query = query.astype("float32")
        distances, indices = index.search(query, top_k)

        results = {
            "query_names": query_names or [f"query_{i}" for i in range(indices.shape[0])],
            "image_names": [],
            "similarities": [],
            "labels": [],
        }

        for row_idx, (row_inds, row_dist) in enumerate(zip(indices, distances)):
            image_names = []
            sims = []
            row_labels = []
            for idx, dist in zip(row_inds, row_dist):
                if idx < 0:
                    continue
                image_names.append(metadata[idx])
                sims.append(float(dist))
                row_labels.append(labels[idx])
            results["image_names"].append(image_names)
            results["similarities"].append(sims)
            results["labels"].append(row_labels)

        return results

    def to_dataframe(self, results: Dict) -> pd.DataFrame:
        rows = []
        for query_name, image_names, sims, labels in zip(
            results["query_names"], results["image_names"], results["similarities"], results["labels"]
        ):
            for image_name, sim, label in zip(image_names, sims, labels):
                rows.append(
                    {
                        "query_name": query_name,
                        "image_name": image_name,
                        "similarity": np.round(sim, 6),
                        "label": label,
                    }
                )
        return pd.DataFrame(rows)
