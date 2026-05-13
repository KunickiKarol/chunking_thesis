

import json


def find_query_by_id(task_input_dir, query_key):
    for search_file in task_input_dir.glob("*.json"):
        with open(search_file, "r", encoding="utf-8") as f:
            search_data = json.load(f)
        if query_key in search_data:
            return search_data[query_key]
    return None

def find_retreived_chunk_by_id(chunks_input_dir,
                        embed_input_dir,
                        retrieved_key,
                        split
                    ):
    for chunk_file in chunks_input_dir.glob("*.json"):
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)
        if chunk_data.get("id") == retrieved_key:
            return chunk_data
    return None
