import importlib
import pkgutil

from src.generation.prompts.register import PROMPTS


def load_prompt():
    """
    Auto-import wszystkich modułów w folderze prompts/
    żeby dekoratory @register_prompt się odpaliły.
    """

    package = "src.generation.prompts"
    for _, module_name, _ in pkgutil.iter_modules(importlib.import_module(package).__path__):
        if module_name in {"register", "all_generation"}:
            continue

        importlib.import_module(f"{package}.{module_name}")


# init side-effect
load_prompt()


def generate_prompt(prompt_name: str, question_text, question_options, are_options, chunks):
    try:
        fn = PROMPTS[prompt_name]
    except KeyError:
        raise ValueError(f"Unknown prompt type: {prompt_name}")
    return fn(question_text, question_options, are_options, chunks)
