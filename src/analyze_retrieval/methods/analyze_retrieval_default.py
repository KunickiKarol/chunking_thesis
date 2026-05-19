
import math
import time
from typing import Any

from src.analyze_retrieval.methods.register import register_analyze_retrieval


@register_analyze_retrieval("default")
def analyze_retrieval_default(
    rerank_results: dict[str, list[str]],
    tasks: dict[str, dict[str, Any]],
    all_chunks: dict[str, dict[str, Any]],
    analyze_preset_params: dict[str, Any],
) -> dict:
    """
    Compute retrieval metrics for reranked results.
 
    Args:
        rerank_results: {question_id: [chunk_id, ...]} — ranked list of retrieved chunk IDs
        tasks:          {question_id: {"source_file": str, ...}}
        all_chunks:     {chunk_id: {"chunk_id": str, "source_file": str, ...}}
        analyze_preset_params: additional parameters (unused in default preset)
 
    Returns:
        {"metrics": {"Accuracy@1": ..., "Accuracy@1-5": [...], "Recall@10": ..., "MRR@10": ..., "NDCG@10": ...}}
    """
    K = 10
    start_time = time.perf_counter()
    def is_relevant(chunk_id: str, question_id: str) -> bool:
        """A chunk is relevant when its source_file matches the question's source_file."""
        chunk = all_chunks.get(chunk_id)
        task = tasks.get(question_id)
        if chunk is None or task is None:
            return False
        return chunk.get("source_file") == task.get("source_id")
 
    acc_at_1 = 0.0
    acc_at_k = [0.0] * 5       # positions 1-5
    recall_at_10_total = 0.0
    mrr_total = 0.0
    ndcg_total = 0.0
 
    n = len(tasks)
    if n == 0:
        return {
            "metrics": {
                "Accuracy@1": 0.0,
                "Accuracy@1-5": [0.0] * 5,
                "Recall@10": 0.0,
                "MRR@10": 0.0,
                "NDCG@10": 0.0,
            }
        }
 
    for question_id in tasks:
        retrieved = rerank_results.get(question_id, [])
        top_k = retrieved[:K]
 
        relevance = [is_relevant(cid, question_id) for cid in top_k]
 
        # ── Accuracy@1 ────────────────────────────────────────────────────────
        if relevance and relevance[0]:
            acc_at_1 += 1.0
 
        # ── Accuracy@1-5 (cumulative: hit within top-i for i in 1..5) ─────────
        hit = False
        for i in range(5):
            if i < len(relevance) and relevance[i]:
                hit = True
            if hit:
                acc_at_k[i] += 1.0
 
        # ── Recall@10 ─────────────────────────────────────────────────────────
        # Total relevant chunks for this question (across the whole corpus)
        total_relevant = sum(
            1 for cid in all_chunks if is_relevant(cid, question_id)
        )
        hits_in_top_k = sum(1 for r in relevance if r)
        if total_relevant > 0:
            recall_at_10_total += hits_in_top_k / total_relevant
        else:
            recall_at_10_total += 0.0
 
        # ── MRR@10 ────────────────────────────────────────────────────────────
        for rank, rel in enumerate(relevance, start=1):
            if rel:
                mrr_total += 1.0 / rank
                break
 
        # ── NDCG@10 ───────────────────────────────────────────────────────────
        # Binary relevance: DCG = sum(rel_i / log2(i+1))
        dcg = sum(
            (1.0 / math.log2(rank + 1))
            for rank, rel in enumerate(relevance, start=1)
            if rel
        )
        # Ideal DCG: all relevant docs ranked first
        ideal_hits = min(hits_in_top_k, K)  # can't have more hits than retrieved
        # For binary relevance, IDCG = sum_{i=1}^{ideal_hits} 1/log2(i+1)
        # but we also need to account for total_relevant if < ideal_hits
        ideal_count = min(total_relevant, K) if total_relevant > 0 else hits_in_top_k
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_count + 1))
        if idcg > 0:
            ndcg_total += dcg / idcg
 
    metrics = {
        "Accuracy@1": round(acc_at_1 / n, 4),
        "Accuracy@1-5": [round(acc_at_k[i] / n, 4) for i in range(5)],
        "Recall@10": round(recall_at_10_total / n, 4),
        "MRR@10": round(mrr_total / n, 4),
        "NDCG@10": round(ndcg_total / n, 4),
    }
    total_time = time.perf_counter() - start_time
    return {"metrics": metrics}, total_time
