from .geolocalization_evaluator import GeoLocalizationEvaluator


def build_evaluator(args, gt_csv_path, radius_deg: float, max_k: int):
    if gt_csv_path is None:
        return None
    task_name = getattr(args, "task_name", None) or "global_geolocalization"
    if task_name == "global_geolocalization":
        return GeoLocalizationEvaluator(gt_csv_path=gt_csv_path, radius_deg=radius_deg, max_k=max_k)
    raise ValueError(f"Unsupported task_name for evaluator build: {task_name}")
