from typing import Callable, Dict, List

RerankerFn = Callable[[str], List[str]]

RERANKERS: Dict[str, RerankerFn] = {}


def register_reranker(name: str):
    def decorator(fn: RerankerFn):
        RERANKERS[name] = fn
        return fn

    return decorator
