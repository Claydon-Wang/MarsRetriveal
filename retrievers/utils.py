from .faiss_retriever import FaissRetriever


def build_retriever(args, database):
    return FaissRetriever(args=args, database=database)
