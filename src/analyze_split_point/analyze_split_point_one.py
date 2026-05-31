import json

from src.analyze_split_point.methods.all_analyze_split_point import analyze_split_point


def analyze_split_point_one(
    analyze_type,
    analyze_preset_params,
    chunks_input_dir,
    books_input_dir,
    tags_input_dir,
    result_dir,
):
    chunks_files = list(chunks_input_dir.glob("*.jsonl"))
    books_files = list(books_input_dir.glob("*.txt"))
    tags_files = list(tags_input_dir.glob("*.json"))

    results_metrics, time = analyze_split_point(
        analyze_type, chunks_files, books_files, tags_files, analyze_preset_params
    )

    with open(result_dir / "analyze_split_point.json", "w", encoding="utf-8") as f:
        json.dump(results_metrics, f, ensure_ascii=False, indent=4)

    with open(result_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"analyze_split_point_time": time}, f, ensure_ascii=False, indent=4)

    return results_metrics, time
