import json
import time
from pathlib import Path

from src.tools.find_content import (
    find_query_by_id,
    find_retreived_chunk_by_id
)
from src.rerank.methods.all_reranker import rerank_text


def rerank_one(
    rerank_name,
    rerank_preset_params,
    split,
    task_input_dir: Path,
    chunks_input_dir: Path,
    embed_input_dir: Path,
    search_input_dir: Path,
    result_dir,
):
    search_dir = search_input_dir / "Search"

    # cache na query i chunki
    query_cache = {}
    chunk_cache = {}

    # nowy format:
    # result[query_key] = (
    #     question_text,
    #     [(retrieved_key, chunk_text), ...]
    # )
    result = {}

    global_rerank_time = 0.0

    for search_file in search_dir.glob("*.json"):
        with open(search_file, "r", encoding="utf-8") as f:
            search_data = json.load(f)

        for query_key, query_data in search_data.items():

            # cache query
            if query_key not in query_cache:
                query_cache[query_key] = find_query_by_id(
                    task_input_dir,
                    query_key
                )["Question"]

            question_text = query_cache[query_key]

            retrieved_chunks = []

            for retrieved_key in query_data.get("indices", []):

                # cache chunków
                if retrieved_key not in chunk_cache:
                    chunk_cache[retrieved_key] = find_retreived_chunk_by_id(
                        chunks_input_dir,
                        embed_input_dir,
                        retrieved_key,
                        split
                    )["text"]

                chunk_text = chunk_cache[retrieved_key]

                retrieved_chunks.append(
                    {retrieved_key: chunk_text}
                )

            # nowy słownik
            result[query_key] = (
                question_text,
                retrieved_chunks
            )

    # reranking + pomiar czasu
    rerank_results = {}

    for query_key, query_tuple in result.items():

        start_time = time.perf_counter()

        reranked = rerank_text(
            rerank_name,
            query_tuple,
            **rerank_preset_params
        )

        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        global_rerank_time += elapsed_time

        rerank_results[query_key] = reranked

    # zapis wyników rerankingu
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    rerank_output_path = result_dir / "rerank_results.json"

    with open(rerank_output_path, "w", encoding="utf-8") as f:
        json.dump(rerank_results, f, ensure_ascii=False, indent=2)

    # zapis meta
    meta = {
        "global_rerank_time": global_rerank_time
    }

    meta_output_path = result_dir / "meta.json"

    with open(meta_output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)