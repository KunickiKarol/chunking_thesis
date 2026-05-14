import importlib
import pkgutil

from src.generation.methods.register import GENERATORS


def load_generator():
    """
    Auto-import wszystkich modułów w folderze methods/
    żeby dekoratory @register_generator się odpaliły.
    """

    package = "src.generation.methods"
    for _, module_name, _ in pkgutil.iter_modules(importlib.import_module(package).__path__):
        if module_name in {"register", "all_generation"}:
            continue

        importlib.import_module(f"{package}.{module_name}")


# init side-effect
load_generator()


def generate_text(generator_name: str, rerank_results, tasks, all_chunks, generation_preset_params):
    try:
        fn = GENERATORS[generator_name]
    except KeyError:
        raise ValueError(f"Unknown generator type: {generator_name}")
    return fn(rerank_results, tasks, all_chunks, generation_preset_params)
