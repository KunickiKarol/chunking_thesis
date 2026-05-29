from src.rerank.methods.register import register_reranker
from src.tools.models_cache import get_reranker


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

    top_k = rerank_preset_params.get("top_k")
    order_preverse = rerank_preset_params.get("order_preverse")
    reranker_name = rerank_preset_params.get("reranker_name")

    question_text = query_tuple[0]
    retrieved_chunks = query_tuple[1]

    if not retrieved_chunks:
        return [], []

    reranker = get_reranker(reranker_name)

    pairs = []
    metadata = []

    for chunk in retrieved_chunks:
        key = chunk[0]
        text = chunk[1]

        metadata.append(key)
        pairs.append([question_text, text])

    scores = reranker.compute_score(pairs)

    ranked = sorted(zip(metadata, scores), key=lambda x: x[1], reverse=True)

    ranked = ranked[:top_k]

    if order_preverse:
        ranked = sorted(ranked, key=lambda x: x[0])

    ranked_keys = [key for key, _ in ranked]
    ranked_scores = [score for _, score in ranked]

    return ranked_keys, ranked_scores
