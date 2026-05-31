import json
from pathlib import Path


def find_query_by_id(task_input_dir, query_key):
    for search_file in task_input_dir.glob("*.json"):
        with open(search_file, "r", encoding="utf-8") as f:
            search_data = json.load(f)
        if query_key in search_data:
            return search_file.stem, search_data[query_key]
    raise ValueError(f"Nie znaleziono query o query_key={query_key} w {task_input_dir}")


def find_chunk_by_retreived_id(embed_input_dir, retrieved_key, chunks_input_dir, source_id=None):
    source_id, chunk_id = find_chunk_id_by_retreived_id(embed_input_dir, retrieved_key, source_id)
    return find_chunk_by_id(chunks_input_dir, source_id, chunk_id)


def find_chunk_id_by_retreived_id(embed_input_dir, retrieved_key, source_id=None):
    metadata_dir = embed_input_dir / "Metadatas"
    for path in metadata_dir.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            for meta in json.load(f):
                if meta.get("id") == retrieved_key and (source_id is None or meta.get("source_file") == source_id):
                    return source_id if source_id else meta.get("source_file"), meta.get("chunk_id")
    raise ValueError(f"Nie znaleziono chunku o source_id={source_id} i retrieved_key={retrieved_key}")


def find_chunk_by_id(chunks_input_dir, source_id, chunk_id):
    for chunk_file in chunks_input_dir.glob(f"{source_id}.jsonl"):

        with open(chunk_file, "r", encoding="utf-8") as f:

            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunk_data = json.loads(line)
                if chunk_data.get("chunk_id") == chunk_id:
                    return chunk_data

    raise ValueError(f"Nie znaleziono chunku o source_id={source_id} i chunk_id={chunk_id}")


def load_all_queries_fast(task_input_dir: Path) -> dict:
    """{ query_key -> (source_id, query_data) }"""
    result = {}
    for path in task_input_dir.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            for query_key, query_data in json.load(f).items():
                result[query_key] = (path.stem, query_data)
    return result


def load_all_queries_fast_source(task_input_dir: Path) -> dict:
    """{ source_id -> [(query_key, query_data)] }"""
    result = {}
    for path in task_input_dir.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            for query_key, query_data in json.load(f).items():
                if path.stem not in result:
                    result[path.stem] = []
                result[path.stem].append((query_key, query_data))
    return result


def load_all_chunk_metadata_fast(embed_input_dir: Path) -> dict:
    """{ source_id: -> {retrieved_key: chunk_id} }"""
    result = {}
    metadata_dir = embed_input_dir / "Metadatas"
    for path in metadata_dir.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            for meta in json.load(f):
                rid = meta.get("id")
                if meta.get("source_file") not in result:
                    result[meta.get("source_file")] = {}
                result[meta.get("source_file")][rid] = meta.get("chunk_id")
    return result


def load_all_chunk_metadata_fast_global(embed_input_dir: Path) -> dict:
    """{ source_id: -> {retrieved_key: chunk_id} }"""
    result = {}
    result["global"] = {}
    metadata_dir = embed_input_dir / "Metadatas"
    for path in metadata_dir.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            for meta in json.load(f):
                rid = meta.get("id")
                result["global"][rid] = meta.get("chunk_id")
    return result


def load_all_chunks_fast(chunks_input_dir: Path) -> dict:
    """{ chunk_id -> chunk_data }"""
    result = {}
    for path in chunks_input_dir.glob("*.jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunk_data = json.loads(line)
                cid = chunk_data.get("chunk_id")
                if cid is not None:
                    result[cid] = chunk_data
    return result

def load_all_chunks_meta_fast(chunks_input_dir: Path) -> dict:
    with open(chunks_input_dir / "meta.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "total_chunking_time": data.get("total_chunking_time")
    }


# ─── Szybkie wyszukiwania (O(1) zamiast O(n * pliki)) ─────────────────────────


def find_query_by_id_fast(query_index: dict, query_key: str):
    try:
        return query_index[query_key]  # (source_id, query_data)
    except KeyError:
        raise ValueError(f"Nie znaleziono query o query_key={query_key}")


def find_chunk_by_retrieved_id_fast(
    metadata_index: dict,
    chunk_index: dict,
    retrieved_key: int,
    source_id: str | None = None,
):
    source_metadata = metadata_index.get(source_id)
    if not source_metadata:
        raise ValueError(f"Nie znaleziono metadanych dla source_id={source_id}")

    chunk_id = source_metadata.get(retrieved_key)
    if chunk_id is None:
        raise ValueError(f"Nie znaleziono chunku o retrieved_key={retrieved_key} dla source_id={source_id}")

    chunk = chunk_index.get(chunk_id)
    if chunk is None:
        raise ValueError(f"Nie znaleziono chunku {chunk_id} w chunk_index")
    return chunk
