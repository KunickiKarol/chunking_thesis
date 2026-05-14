import json

from src.generation.methods.all_generation import generate_text


def generation_one(
    generation_name,
    generation_preset_params,
    split,
    task_input_dir,
    chunks_input_dir,
    rerank_input_dir,
    result_dir,
):
    with open(rerank_input_dir / "rerank_results.json", "r", encoding="utf-8") as f:
        rerank_results = json.load(f)
    chunks_used = {x for values in rerank_results.values() for x in values}

    tasks = {}
    for task_file in task_input_dir.glob("*.json"):
        with open(task_file, "r", encoding="utf-8") as f:
            task_data = json.load(f)
        tasks.update(task_data)

    all_chunks = {}
    for chunk_file in chunks_input_dir.glob("*.jsonl"):
        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in f:
                chunk_data = json.loads(line)
                chunk_id = chunk_data["chunk_id"]
                if chunk_id in chunks_used:
                    all_chunks[chunk_id] = chunk_data["text"]

    answers, time = generate_text(generation_name, rerank_results, tasks, all_chunks, generation_preset_params)

    with open(result_dir / "generation_results.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)

    with open(result_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"generation_time": time}, f, ensure_ascii=False, indent=4)
