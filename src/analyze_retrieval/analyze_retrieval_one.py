import json

from src.analyze_retrieval.methods.all_analyze_retrieval import analyze_retrieval



def analyze_retrieval_one(
    analyze_retrieval_name,
    analyze_retrieval_preset_params,
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
        for key, value in task_data.items():
            value["source_id"] = task_file.stem
        tasks.update(task_data)

    all_chunks = {}
    for chunk_file in chunks_input_dir.glob("*.jsonl"):
        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in f:
                chunk_data = json.loads(line)
                chunk_id = chunk_data["chunk_id"]
                if chunk_id in chunks_used:
                    all_chunks[chunk_id] = {'source_file': chunk_data["source_file"], 'chunk_id': chunk_id}

    answers, time = analyze_retrieval(analyze_retrieval_name, rerank_results, tasks, all_chunks, analyze_retrieval_preset_params)

    with open(result_dir / "analyze_retrieval_results.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)

    with open(result_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"analyze_retrieval_time": time}, f, ensure_ascii=False, indent=4)
