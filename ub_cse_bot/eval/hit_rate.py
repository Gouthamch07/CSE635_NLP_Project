from __future__ import annotations

from pathlib import Path

from ..rag.hybrid import HybridRetriever
from ..utils.io import read_jsonl
from ..utils.logging import get_logger

log = get_logger(__name__)


def hit_rate_at_k(
    retriever: HybridRetriever,
    dataset_path: Path,
    k: int = 5,
) -> dict:
    """Hit@K and MRR@K.

    Dataset schema (jsonl):
        {"question": "...", "relevant_urls": ["https://...", ...]}
    """
    rows = read_jsonl(dataset_path)
    hits = 0
    rr_sum = 0.0
    for row in rows:
        q = row["question"]
        gold = set(row.get("relevant_urls") or [])
        trace = retriever.retrieve(q, final_k=k)
        retrieved = [h.url for h in trace.hits[:k]]
        hit_rank = next((i for i, u in enumerate(retrieved) if u in gold), None)
        if hit_rank is not None:
            hits += 1
            rr_sum += 1.0 / (hit_rank + 1)
    n = max(1, len(rows))
    metrics = {"hit@k": hits / n, "mrr@k": rr_sum / n, "n": len(rows), "k": k}
    log.info("retrieval metrics %s", metrics)
    return metrics
