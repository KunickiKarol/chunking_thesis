from typing import Callable, Dict, List

AnalyzeEmbeddingsFn = Callable[[str], List[str]]

ANALYZE_EMBEDDINGS: Dict[str, AnalyzeEmbeddingsFn] = {}


def register_analyze_embeddings(name: str):
    def decorator(fn: AnalyzeEmbeddingsFn):
        ANALYZE_EMBEDDINGS[name] = fn
        return fn

    return decorator
