import json


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
                if meta.get("id") == retrieved_key and (source_id is None or meta.get("source") == source_id):
                    return source_id if source_id else meta.get("source"), meta.get("chunk_id")
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
