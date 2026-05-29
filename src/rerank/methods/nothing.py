from src.rerank.methods.register import register_reranker


@register_reranker("nothing")
def rank_nothing(query_tuple, **rerank_preset_params):
    top_k = rerank_preset_params.get("top_k")
    order_preverse = rerank_preset_params.get("order_preverse")

    # query_tuple = (
    #     question_text,
    #     [(retrieved_key, chunk_text), ...]
    # )

    retrieved_chunks = query_tuple[1]

    # bierzemy top_k
    sliced = retrieved_chunks[:top_k]

    # klucze
    keys = [x[0] for x in sliced]

    # sztuczne score (brak rerankingu -> neutralne 1.0)
    scores = [1.0 for _ in keys]

    if order_preverse:
        # sortujemy razem, żeby nie rozjechały się listy
        combined = sorted(zip(keys, scores), key=lambda x: x[0])
        keys, scores = zip(*combined) if combined else ([], [])
        keys = list(keys)
        scores = list(scores)

    return keys, scores
