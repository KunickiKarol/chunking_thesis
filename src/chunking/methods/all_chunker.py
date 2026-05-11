import pkgutil
import importlib
from src.chunking.methods.register import CHUNKERS


def load_chunkers():
    """
    Auto-import wszystkich modułów w folderze methods/
    żeby dekoratory @register_chunker się odpaliły.
    """

    package = "src.chunking.methods"
    for _, module_name, _ in pkgutil.iter_modules(
        importlib.import_module(package).__path__
    ):
        if module_name in {"register", "all_chunker"}:
            continue

        importlib.import_module(f"{package}.{module_name}")


# init side-effect
load_chunkers()


def chunk_text(chunking_type: str, text: str, **params):
    try:
        fn = CHUNKERS[chunking_type]
    except KeyError:
        raise ValueError(f"Unknown chunking type: {chunking_type}")

    return fn(text, **params)