import json
import logging
import time
from pathlib import Path

from src.rerank.methods.all_reranker import rerank_text
from src.tools.find_content import find_query_by_id_fast, load_all_chunks_fast, load_all_queries_fast

logger = logging.getLogger(__name__)


def rerank_one(
    rerank_name,
    rerank_preset_params,
    task_input_dir: Path,
    chunks_input_dir: Path,
    embed_input_dir: Path,
    search_input_dir: Path,
    result_dir,
):
    # Wczytaj wszystko raz
    query_index = load_all_queries_fast(task_input_dir)
    chunk_index = load_all_chunks_fast(chunks_input_dir)

    search_dir = search_input_dir / "Search"
    result = {}
    global_rerank_time = 0.0

    for search_file in search_dir.glob("*.json"):
        with open(search_file, encoding="utf-8") as f:
            search_data = json.load(f)

        for query_key, query_data in search_data.items():
            _, question_data = find_query_by_id_fast(query_index, query_key)
            question_text = question_data["Question"]

            retrieved_chunks = []
            for retrieved_key in query_data.get("chunks"):
                chunk = chunk_index[retrieved_key]
                retrieved_chunks.append((chunk["chunk_id"], chunk["text"]))

            result[query_key] = (question_text, retrieved_chunks)

    # reranking + pomiar czasu
    rerank_results = {}
    scores_results = {}

    for query_key, query_tuple in result.items():

        start_time = time.perf_counter()

        reranked, scores = rerank_text(rerank_name, query_tuple, **rerank_preset_params)

        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        global_rerank_time += elapsed_time

        rerank_results[query_key] = reranked
        scores_results[query_key] = scores

    # zapis wyników rerankingu
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    rerank_output_path = result_dir / "rerank_results.json"
    scores_output_path = result_dir / "scores_results.json"
    with open(rerank_output_path, "w", encoding="utf-8") as f:
        json.dump(rerank_results, f, ensure_ascii=False, indent=2)

    with open(scores_output_path, "w", encoding="utf-8") as f:
        json.dump(scores_results, f, ensure_ascii=False, indent=2)

    # zapis meta
    meta = {"global_rerank_time": global_rerank_time}

    meta_output_path = result_dir / "meta.json"

    with open(meta_output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
