from typing import Callable, Dict, List

EmbedMetricsFn = Callable[[str], List[str]]

EMBED_METRICS: Dict[str, EmbedMetricsFn] = {}


def register_embed_metric(name: str):
    def decorator(fn: EmbedMetricsFn):
        EMBED_METRICS[name] = fn
        return fn

    return decorator
