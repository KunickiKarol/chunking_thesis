from typing import Callable, Dict, List

ChunkerFn = Callable[[str], List[str]]

CHUNKERS: Dict[str, ChunkerFn] = {}


def register_chunker(name: str):
    def decorator(fn: ChunkerFn):
        CHUNKERS[name] = fn
        return fn
    return decorator
