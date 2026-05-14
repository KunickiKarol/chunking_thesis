from typing import Callable, Dict, List

PromptFn = Callable[[str], List[str]]

PROMPTS: Dict[str, PromptFn] = {}


def register_prompt(name: str):
    def decorator(fn: PromptFn):
        PROMPTS[name] = fn
        return fn

    return decorator
