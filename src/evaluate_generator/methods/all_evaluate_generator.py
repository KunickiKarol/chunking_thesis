import importlib
import pkgutil

from src.evaluate_generator.methods.register import EVALUATORS_GENERATORS


def load_generator():
    """
    Auto-import wszystkich modułów w folderze methods/
    żeby dekoratory @register_evaluator_generator się odpaliły.
    """

    package = "src.evaluate_generator.methods"
    for _, module_name, _ in pkgutil.iter_modules(importlib.import_module(package).__path__):
        if module_name in {"register", "all_evaluate_generator"}:
            continue

        importlib.import_module(f"{package}.{module_name}")


# init side-effect
load_generator()


def evaluate_generator(generator_name: str, generation_results, tasks, evaluation_preset_params):
    try:
        fn = EVALUATORS_GENERATORS[generator_name]
    except KeyError:
        raise ValueError(f"Unknown generator type: {generator_name}")
    return fn(generation_results, tasks, evaluation_preset_params)
