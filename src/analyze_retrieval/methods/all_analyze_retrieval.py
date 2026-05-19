import importlib
import pkgutil

from src.analyze_retrieval.methods.register import ANALYZE_RETRIEVAL


def load_analyze_retrieval():
    """
    Auto-import wszystkich modułów w folderze methods/
    żeby dekoratory @register_analyze_retrieval się odpaliły.
    """

    package = "src.analyze_retrieval.methods"
    for _, module_name, _ in pkgutil.iter_modules(importlib.import_module(package).__path__):
        if module_name in {"register", "all_analyze_retrieval"}:
            continue

        importlib.import_module(f"{package}.{module_name}")


# init side-effect
load_analyze_retrieval()


def analyze_retrieval(analyzer_type: str, chunks_files, books_files, tags_files, analyze_preset_params):
    try:
        fn = ANALYZE_RETRIEVAL[analyzer_type]
    except KeyError:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")
    return fn(chunks_files, books_files, tags_files, analyze_preset_params)
