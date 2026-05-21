import importlib
import pkgutil

from src.analyze_embeddings.methods.register import ANALYZE_EMBEDDINGS



def load_analyze_embeddings():
    """
    Auto-import wszystkich modułów w folderze methods/
    żeby dekoratory @register_analyze_embeddings się odpaliły.
    """

    package = "src.analyze_embeddings.methods"
    for _, module_name, _ in pkgutil.iter_modules(importlib.import_module(package).__path__):
        if module_name in {"register", "all_analyze_embeddings"}:
            continue

        importlib.import_module(f"{package}.{module_name}")


# init side-effect
load_analyze_embeddings()


def get_analyze_embeddings(analyze_embedding_name: str, analyze_preset_params, result_dir, df_embedding, df_bookmeta):
    try:
        fn = ANALYZE_EMBEDDINGS[analyze_embedding_name]
    except KeyError:
        raise ValueError(f"Unknown analyze embeddings type: {analyze_embedding_name}")
    return fn(analyze_preset_params, result_dir, df_embedding, df_bookmeta)
