from .geolocalization_retriever import GeoLocalizationRetriever


def build_retriever(args, database):
    task_name = getattr(args, "task_name", None) or "global_geolocalization"
    if task_name == "global_geolocalization":
        return GeoLocalizationRetriever(args=args, database=database)
    raise ValueError(f"Unsupported task_name for retriever build: {task_name}")
