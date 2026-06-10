# src/search/search_query.py

import json
import time
from pathlib import Path

import faiss
import numpy as np

from src.tools.find_content import (
    load_all_chunk_metadata_fast,
    load_all_chunk_metadata_fast_global,
    load_all_queries_fast_source,
)
from src.tools.models_cache import get_sentence_transformer


def search_query(
    embed_type: str,
    search_params: dict,
    task_input_dir: Path,
    chunks_input_dir: Path,
    embed_input_dir: Path,
    result_dir: Path,
):
    top_k = search_params.get("top_k", 5)
    embedder_name = search_params.get("embbeder")

    _search(embed_type, embedder_name, top_k, task_input_dir, chunks_input_dir, embed_input_dir, result_dir)


def _search(
    embed_type: str,
    embedder_name: str,
    top_k: int,
    task_input_dir: Path,
    chunks_input_dir: Path,
    embed_input_dir: Path,
    result_dir: Path,
):
    model = get_sentence_transformer(embedder_name)
    index_path = embed_input_dir / "Indexes"
    query_index = load_all_queries_fast_source(task_input_dir)
    if embed_type == "global":
        metadata_index = load_all_chunk_metadata_fast_global(embed_input_dir)
    else:
        metadata_index = load_all_chunk_metadata_fast(embed_input_dir)

    if not index_path.exists():
        raise FileNotFoundError(f"Brak pliku indeksu FAISS: {index_path}")

    index_files = list(index_path.rglob("*.index"))
    total_time = 0.0
    for index_file in index_files:
        index = faiss.read_index(str(index_file))
        results_global = {}

        if embed_type == "global":
            tasks = [task for task_list in query_index.values() for task in task_list]
        else:
            tasks = query_index.get(index_file.stem)
        results = {}
        for task_key, task_data in tasks:

            question_text = task_data.get("Question")

            # Embed the question
            t_start = time.perf_counter()
            query_vector = model.encode([question_text], convert_to_numpy=True, show_progress_bar=False).astype(
                np.float32
            )
            t_after_embed = time.perf_counter()

            # Search in FAISS index
            distances, indices = index.search(query_vector, top_k)
            t_end = time.perf_counter()

            elapsed = t_end - t_start
            total_time += elapsed
            chunks = [metadata_index[index_file.stem].get(indice) for indice in indices[0] if indice != -1]
            results[task_key] = {
                "chunks": chunks,
                "distances": distances[0].tolist(),
                "embed_time": t_after_embed - t_start,
                "search_time": t_end - t_after_embed,
            }
        results_global.update(results)
        if embed_type == "local":
            search_out_dir = result_dir / "Search"
            search_out_dir.mkdir(parents=True, exist_ok=True)

            with open(search_out_dir / f"{index_file.stem}.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    if embed_type == "global":
        search_out_dir = result_dir / "Search"
        search_out_dir.mkdir(parents=True, exist_ok=True)

        with open(search_out_dir / f"global.json", "w", encoding="utf-8") as f:
            json.dump(results_global, f, ensure_ascii=False, indent=2)

    # Save metadata
    meta = {"total_time": total_time}
    with open(result_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
