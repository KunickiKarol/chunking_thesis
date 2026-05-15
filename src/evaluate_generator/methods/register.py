from typing import Callable, Dict, List

EvaluatorGeneratorFn = Callable[[str], List[str]]

EVALUATORS_GENERATORS: Dict[str, EvaluatorGeneratorFn] = {}


def register_evaluator_generator(name: str):
    def decorator(fn: EvaluatorGeneratorFn):
        EVALUATORS_GENERATORS[name] = fn
        return fn

    return decorator
