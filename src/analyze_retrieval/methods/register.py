from typing import Callable, Dict, List

AnalyzeRetrievalFn = Callable[[str], List[str]]

ANALYZE_RETRIEVAL: Dict[str, AnalyzeRetrievalFn] = {}


def register_analyze_retrieval(name: str):
    def decorator(fn: AnalyzeRetrievalFn):
        ANALYZE_RETRIEVAL[name] = fn
        return fn

    return decorator
