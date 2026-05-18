from typing import Callable, Dict, List

AnalyzeSplitPointFn = Callable[[str], List[str]]

ANALYZE_SPLIT_POINTS: Dict[str, AnalyzeSplitPointFn] = {}


def register_analyze_split_point(name: str):
    def decorator(fn: AnalyzeSplitPointFn):
        ANALYZE_SPLIT_POINTS[name] = fn
        return fn

    return decorator
