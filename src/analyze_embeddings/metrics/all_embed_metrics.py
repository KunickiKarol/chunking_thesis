import importlib
import pkgutil

from src.analyze_embeddings.metrics.register import EMBED_METRICS


def load_embed_metric():
    """
    Auto-import wszystkich modułów w folderze methods/
    żeby dekoratory @register_embed_metric się odpaliły.
    """

    package = "src.analyze_embeddings.metrics"
    for _, module_name, _ in pkgutil.iter_modules(importlib.import_module(package).__path__):
        if module_name in {"register", "all_embed_metrics"}:
            continue

        importlib.import_module(f"{package}.{module_name}")


# init side-effect
load_embed_metric()


def get_metric(embed_metric_name: str, rerank_results, tasks, all_chunks, generation_preset_params):
    try:
        fn = EMBED_METRICS[embed_metric_name]
    except KeyError:
        raise ValueError(f"Unknown embed metric type: {embed_metric_name}")
    return fn(rerank_results, tasks, all_chunks, generation_preset_params)
