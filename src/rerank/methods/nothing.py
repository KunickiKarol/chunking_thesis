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
    retrieved_chunks = [list(x.keys())[0] for x in retrieved_chunks[:top_k]]
    if order_preverse:
        retrieved_chunks = sorted(retrieved_chunks)
    return retrieved_chunks
