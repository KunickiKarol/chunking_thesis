import importlib
import pkgutil

from src.analyze_split_point.methods.register import ANALYZE_SPLIT_POINTS


def load_generator():
    """
    Auto-import wszystkich modułów w folderze methods/
    żeby dekoratory @register_analyze_split_point się odpaliły.
    """

    package = "src.analyze_split_point.methods"
    for _, module_name, _ in pkgutil.iter_modules(importlib.import_module(package).__path__):
        if module_name in {"register", "all_analyze_split_points"}:
            continue

        importlib.import_module(f"{package}.{module_name}")


# init side-effect
load_generator()


def analyze_split_point(analyzer_type: str, chunks_files, books_files, tags_files, analyze_preset_params):
    try:
        fn = ANALYZE_SPLIT_POINTS[analyzer_type]
    except KeyError:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")
    return fn(chunks_files, books_files, tags_files, analyze_preset_params)
