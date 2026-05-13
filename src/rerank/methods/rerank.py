from threading import Lock

from FlagEmbedding import FlagReranker

from src.rerank.methods.register import register_reranker

# Globalny cache singletonów
_RERANKER_MODELS = {}
_RERANKER_LOCK = Lock()


def get_reranker(model_name: str):
    """
    Singleton/cache dla rerankerów.
    Ładuje model tylko raz per model_name.
    """
    global _RERANKER_MODELS

    if model_name not in _RERANKER_MODELS:
        with _RERANKER_LOCK:
            # double check po locku
            if model_name not in _RERANKER_MODELS:
                _RERANKER_MODELS[model_name] = FlagReranker(model_name, use_fp16=True)

    return _RERANKER_MODELS[model_name]


@register_reranker("reranker")
def rank_bge_reranker(query_tuple, **rerank_preset_params):
    """
    query_tuple = (
        question_text,
        [
            {retrieved_key: chunk_text},
            ...
        ]
    )
    """

    top_k = rerank_preset_params.get("top_k", 10)
    order_preverse = rerank_preset_params.get("order_preverse")

    # np. "BAAI/bge-reranker-v2-m3"
    reranker_name = rerank_preset_params.get("reranker_name", "BAAI/bge-reranker-v2-m3")

    question_text = query_tuple[0]
    retrieved_chunks = query_tuple[1]

    if not retrieved_chunks:
        return []

    reranker = get_reranker(reranker_name)

    # przygotowanie danych do rerankingu
    pairs = []
    metadata = []

    for chunk in retrieved_chunks:
        key = list(chunk.keys())[0]
        text = chunk[key]

        metadata.append(key)
        pairs.append([question_text, text])

    # score dla każdej pary (query, chunk)
    scores = reranker.compute_score(pairs)

    # sortowanie po score malejąco
    ranked = sorted(zip(metadata, scores), key=lambda x: x[1], reverse=True)

    # zwracamy tylko keys
    ranked_keys = [key for key, _ in ranked[:top_k]]
    if order_preverse:
        ranked_keys = sorted(ranked_keys)
    return ranked_keys
