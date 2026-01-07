from .geolocalization_retriever import GeoLocalizationRetriever
from .landform_retriever import LandformRetriever
from .cross_modal_matching_retriever import CrossModalMatchingRetriever


def build_retriever(args, database):
    task_name = getattr(args, "task_name", None) or "global_geolocalization"
    if task_name == "global_geolocalization":
        return GeoLocalizationRetriever(args=args, database=database)
    if task_name == "landform_retrieval":
        return LandformRetriever(args=args, database=database)
    if task_name == "cross_modal_matching":
        return CrossModalMatchingRetriever(args=args, database=database)
    raise ValueError(f"Unsupported task_name for retriever build: {task_name}")
