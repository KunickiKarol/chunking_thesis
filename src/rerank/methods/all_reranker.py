import pkgutil
import importlib
from src.rerank.methods.register import RERANKERS


def load_reranker():
    """
    Auto-import wszystkich modułów w folderze methods/
    żeby dekoratory @register_eranker się odpaliły.
    """

    package = "src.rerank.methods"
    for _, module_name, _ in pkgutil.iter_modules(
        importlib.import_module(package).__path__
    ):
        if module_name in {"register", "all_reranker"}:
            continue

        importlib.import_module(f"{package}.{module_name}")


# init side-effect
load_reranker()


def chunk_text(rerankertype: str, text: str, **params):
    try:
        fn = RERANKERS[rerankertype]
    except KeyError:
        raise ValueError(f"Unknown reranker type: {rerankertype}")

    return fn(text, **params)