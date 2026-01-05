import logging
from typing import Dict, List

import numpy as np

from .base import EvaluatorBase


class LandformRetrievalEvaluator(EvaluatorBase):
    def __init__(self, gt_map: Dict[str, set], max_k: int = None):
        self.gt_map = gt_map or {}
        self.max_k = max_k

    def evaluate(self, pred_df, label: str = "run") -> Dict:
        if pred_df.empty:
            logging.warning("Prediction dataframe is empty, skipping evaluation.")
            return {}

        query_names = sorted(set(pred_df["query_name"].tolist()))
        ap_list: List[float] = []
        recall1_list: List[float] = []
        recall5_list: List[float] = []
        recall10_list: List[float] = []
        precision10_list: List[float] = []

        for q in query_names:
            gt_set = self.gt_map.get(q, set())
            if not gt_set:
                continue
            df_q = pred_df[pred_df["query_name"] == q].sort_values("similarity", ascending=False)
            if self.max_k is not None:
                df_q = df_q.head(self.max_k)
            ranked = df_q["image_name"].tolist()
            hits = 0
            precisions = []
            for i, img in enumerate(ranked, start=1):
                if img in gt_set:
                    hits += 1
                    precisions.append(hits / i)
            ap = sum(precisions) / len(gt_set) if gt_set else 0.0

            top1 = ranked[:1]
            top5 = ranked[:5]
            top10 = ranked[:10]
            recall1 = 1.0 if set(top1) & gt_set else 0.0
            recall5 = 1.0 if set(top5) & gt_set else 0.0
            recall10 = 1.0 if set(top10) & gt_set else 0.0
            precision10 = len(set(top10) & gt_set) / 10.0

            ap_list.append(ap)
            recall1_list.append(recall1)
            recall5_list.append(recall5)
            recall10_list.append(recall10)
            precision10_list.append(precision10)

        metrics = {
            "mAP": float(np.mean(ap_list)) if ap_list else 0.0,
            "recall@1": float(np.mean(recall1_list)) if recall1_list else 0.0,
            "recall@5": float(np.mean(recall5_list)) if recall5_list else 0.0,
            "recall@10": float(np.mean(recall10_list)) if recall10_list else 0.0,
            "precision@10": float(np.mean(precision10_list)) if precision10_list else 0.0,
        }
        logging.info(
            "[%s] mAP=%.4f | Recall@1=%.4f Recall@5=%.4f Recall@10=%.4f Precision@10=%.4f",
            label,
            metrics["mAP"],
            metrics["recall@1"],
            metrics["recall@5"],
            metrics["recall@10"],
            metrics["precision@10"],
        )
        return metrics

    def summary(self, args, args_dynamic, eval_summary: Dict):
        headers = [
            "query_mode",
            "model_name",
            "pretrained",
            "resume_post_train",
            "mAP",
            "recall@1",
            "recall@5",
            "recall@10",
            "precision@10",
        ]
        row = [
            args_dynamic.query_mode,
            args.model,
            args.pretrained,
            args.resume_post_train or "",
            round(eval_summary.get("mAP", 0.0) * 100, 2) if eval_summary else "",
            round(eval_summary.get("recall@1", 0.0) * 100, 2) if eval_summary else "",
            round(eval_summary.get("recall@5", 0.0) * 100, 2) if eval_summary else "",
            round(eval_summary.get("recall@10", 0.0) * 100, 2) if eval_summary else "",
            round(eval_summary.get("precision@10", 0.0) * 100, 2) if eval_summary else "",
        ]
        return headers, row
