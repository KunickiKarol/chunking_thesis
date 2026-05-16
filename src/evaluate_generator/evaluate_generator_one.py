import json

from src.evaluate_generator.methods.all_evaluate_generator import evaluate_generator
from src.generation.methods.all_generation import generate_text


def evaluate_generator_one(
    evaluation_name,
    evaluation_preset_params,
    task_input_dir,
    generation_input_dir,
    result_dir,
):
    with open(generation_input_dir / "generation_results.json", "r", encoding="utf-8") as f:
        generation_results = json.load(f)

    tasks = {}
    for task_file in task_input_dir.glob("*.json"):
        with open(task_file, "r", encoding="utf-8") as f:
            task_data = json.load(f)
        tasks.update(task_data)

    results_metrics, time = evaluate_generator(evaluation_name, generation_results, tasks, evaluation_preset_params)

    with open(result_dir / "generation_results.json", "w", encoding="utf-8") as f:
        json.dump(results_metrics, f, ensure_ascii=False, indent=4)

    with open(result_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"generation_time": time}, f, ensure_ascii=False, indent=4)
