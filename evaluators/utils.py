from .coverage import CoverageEvaluator


def build_evaluator(gt_csv_path, radius_deg: float, max_k: int):
    return CoverageEvaluator(gt_csv_path=gt_csv_path, radius_deg=radius_deg, max_k=max_k)
