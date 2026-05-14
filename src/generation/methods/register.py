from typing import Callable, Dict, List

GeneratorFn = Callable[[str], List[str]]

GENERATORS: Dict[str, GeneratorFn] = {}


def register_generator(name: str):
    def decorator(fn: GeneratorFn):
        GENERATORS[name] = fn
        return fn

    return decorator
