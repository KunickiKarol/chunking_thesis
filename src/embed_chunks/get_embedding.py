import json
from pathlib import Path
from IPython import embed
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def embed_texts(texts, embed_preset_params):
    """
    Tworzy embeddingi dla listy tekstów za pomocą HuggingFace SentenceTransformers.
    """
    model = SentenceTransformer(embed_preset_params["model"])
    embeddings = model.encode(texts, batch_size=embed_preset_params["batch_size"], precision=embed_preset_params["dtype"], show_progress_bar=True)
    return np.array(embeddings, dtype=embed_preset_params["dtype"])


def load_chunks_from_file(file_path: Path):
    """
    Wczytuje wszystkie chunkowane linie z jednego pliku JSONL
    """
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks
    

def build_index_for_file(embed_name, embed_preset_params, input_path, result_dir):
    """
    Tworzy osobny FAISS index dla jednego pliku JSONL
    """
    chunks = load_chunks_from_file(input_path)
    if not chunks:
        print(f"⚠️  Plik {input_path} nie zawiera chunków, pomijam.")
        return

    # embeddingi
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, embed_preset_params)

    # opcjonalnie dodajemy embeddingi do chunków
    for c, e in zip(chunks, embeddings):
        c["embedding"] = e

    
    if embed_name == 'FAISS':
        vector_dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(vector_dim)
    else:
        raise NotImplementedError("Nie zaimplementowano innej vector_db")
    index.add(embeddings)

    # tworzymy ścieżkę wyjściową: ten sam nazwa pliku, inna końcówka .index
    output_file = result_dir / f"{input_path.stem}.index"

    faiss.write_index(index, str(output_file))
    print(f"✅ Zapisano FAISS index dla {input_path.name} w {output_file}, liczba wektorów: {len(chunks)}")

def embed_chunks(embed_name, embed_preset_params, chunks_input_dir, result_dir):
    """
    Tworzy osobny FAISS index dla każdego pliku JSONL w katalogu chunks_input_dir
    """
    files = list(chunks_input_dir.glob("*.jsonl"))
    print(f"Znaleziono {len(files)} plików do przetworzenia w {chunks_input_dir}.")

    for input_path in tqdm(files, desc=f'Tworzenie indexów {result_dir}'):
        build_index_for_file(embed_name, embed_preset_params, input_path, result_dir)