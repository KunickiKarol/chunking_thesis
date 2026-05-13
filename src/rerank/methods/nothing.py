from src.rerank.methods.register import register_reranker


@register_reranker("nothing")
def rank_nothing(
    query_tuple,
    **rerank_preset_params
):
    top_k = rerank_preset_params.get("top_k")

    # query_tuple = (
    #     question_text,
    #     [(retrieved_key, chunk_text), ...]
    # )

    retrieved_chunks = query_tuple[1]

    return [
        retrieved_key
        for retrieved_key, _ in retrieved_chunks[:top_k]
    ]