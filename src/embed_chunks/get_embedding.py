import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_MODELS = {}

import logging

logger = logging.getLogger(__name__)


def get_model(model_name: str):
    if model_name not in _MODELS:
        _MODELS[model_name] = SentenceTransformer(model_name)
    return _MODELS[model_name]


def embed_texts(texts, embed_preset_params):
    model = get_model(embed_preset_params["embbeder"])
    embeddings = model.encode(
        texts,
        batch_size=embed_preset_params["batch_size"],
        precision=embed_preset_params["dtype"],
        show_progress_bar=False,
    )
    return np.array(embeddings, dtype=np.float32)


def load_chunks_from_file(file_path: Path):
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def save_metadata(metadata, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)


def build_index_for_file(embed_name, embeddings, input_path, indexes_dir):
    if embed_name == "FAISS":
        vector_dim = embeddings.shape[1]
        start_index = time.perf_counter()
        index = faiss.IndexFlatL2(vector_dim)
        index.add(embeddings)
        end_index = time.perf_counter()
        index_time = end_index - start_index
    else:
        raise NotImplementedError("Nie zaimplementowano innej vector_db")

    output_file = indexes_dir / f"{input_path.stem}.index"
    faiss.write_index(index, str(output_file))
    return index_time


def embed_chunks(embed_name, embed_preset_params, chunks_input_dir, result_dir):
    result_dir.mkdir(parents=True, exist_ok=True)

    indexes_dir = result_dir / "Indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)

    metadatas_dir = result_dir / "Metadatas"
    metadatas_dir.mkdir(parents=True, exist_ok=True)

    # Nowy folder na embeddingi jako .npy
    embeddings_dir = result_dir / "Embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    files = list(chunks_input_dir.glob("*.jsonl"))

    total_embed_time = 0.0
    total_index_time = 0.0
    total_global_index_time = 0.0

    all_embeddings = []
    global_metadata = []

    embed_type = embed_preset_params.get("embed_type")
    embed_index = 0
    for input_path in files:

        chunks = load_chunks_from_file(input_path)

        if not chunks:
            logger.info(f"⚠️  Plik {input_path} nie zawiera chunków, pomijam.")
            continue

        texts = [c["text"] for c in chunks]

        # ====================================
        # embeddingi
        # ====================================

        start_embed = time.perf_counter()
        embeddings = embed_texts(texts, embed_preset_params)
        end_embed = time.perf_counter()

        embed_time = end_embed - start_embed
        total_embed_time += embed_time

        # ====================================
        # metadata
        # ====================================

        local_metadata = []

        for i, chunk in enumerate(chunks):

            metadata_item = {
                "id": embed_index if embed_type == "global" else i,
                "source_file": chunk["source_file"],
                "chunk_id": chunk["chunk_id"],
                "embed_time": embed_time,
            }
            local_metadata.append(metadata_item)
            if embed_type == "global":
                embed_index += 1
                global_metadata.append(metadata_item)

        # ====================================
        # zapis local: metadata JSON + embeddingi .npy
        # ====================================

        if embed_type == "local":
            # metadata (bez embeddingów)
            metadata_output_file = metadatas_dir / f"{input_path.stem}.json"
            save_metadata(local_metadata, metadata_output_file)

            # embeddingi float32 jako .npy — wiersz i == metadata_item["id"]
            embeddings_output_file = embeddings_dir / f"{input_path.stem}.npy"
            np.save(str(embeddings_output_file), embeddings)
            # embeddings.dtype jest już float32 (wymuszony w embed_texts)

        # ====================================
        # local index — tylko dla trybu "local"
        # ====================================

        if embed_type == "local":
            index_time = build_index_for_file(embed_name, embeddings, input_path, indexes_dir)
            total_index_time += index_time

        # ====================================
        # collect global embeddings — tylko dla trybu "global"
        # ====================================

        if embed_type == "global":
            all_embeddings.append(embeddings)

    # ====================================
    # global index — tylko dla trybu "global"
    # ====================================

    if embed_type == "global" and all_embeddings:

        all_embeddings = np.vstack(all_embeddings)
        vector_dim = all_embeddings.shape[1]

        start_global_index = time.perf_counter()
        global_index = faiss.IndexFlatL2(vector_dim)
        global_index.add(all_embeddings)
        end_global_index = time.perf_counter()

        total_global_index_time = end_global_index - start_global_index

        global_index_path = result_dir / "Indexes" / "global.index"
        faiss.write_index(global_index, str(global_index_path))

        # metadata globalna (bez embeddingów)
        save_metadata(global_metadata, metadatas_dir / "global.json")

        # embeddingi globalne float32 jako .npy — wiersz i == global_metadata[i]["id"]
        # UWAGA: "id" w global_metadata to indeks lokalny w obrębie pliku źródłowego.
        # Globalny indeks wiersza w tablicy = pozycja elementu na liście global_metadata.
        global_embeddings_path = embeddings_dir / "global.npy"
        np.save(str(global_embeddings_path), all_embeddings)

        total_embeddings = int(all_embeddings.shape[0])

    else:
        total_embeddings = 0

    # ====================================
    # meta.json
    # ====================================

    meta = {
        "embed_type": embed_type,
        "embed_time": total_embed_time,
        "index_time": total_global_index_time if embed_type == "global" else total_index_time,
        "total_embeddings": total_embeddings,
    }

    meta_file = result_dir / "meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)
