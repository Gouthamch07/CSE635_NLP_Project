"""Run the full evaluation harness.

Prereqs:
  - Crawl + index have been run.
  - data/eval/retrieval.jsonl and data/eval/ragas.jsonl exist.
"""
from __future__ import annotations

from pathlib import Path

from config import get_settings
from ub_cse_bot.agent import UBCSEAgent
from ub_cse_bot.dialogue import ConversationMemory, PersonalMemory
from ub_cse_bot.eval import (
    benchmark_latency,
    hit_rate_at_k,
    run_ragas,
    run_robustness_suite,
)
from ub_cse_bot.rag.hybrid import HybridRetriever


def main() -> None:
    s = get_settings()
    bm25_path = s.data_dir / "processed" / "bm25.pkl"
    retriever = HybridRetriever.from_disk(bm25_path)
    agent = UBCSEAgent(
        retriever=retriever,
        memory=ConversationMemory(),
        personal=PersonalMemory(Path(s.memory_store_path)),
    )

    out_dir = s.data_dir / "eval_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Retrieval hit-rate
    retr_ds = s.data_dir / "eval" / "retrieval.jsonl"
    if retr_ds.exists():
        hit_rate_at_k(retriever, retr_ds, k=5)

    # RAGAS faithfulness / context precision
    ragas_ds = s.data_dir / "eval" / "ragas.jsonl"
    if ragas_ds.exists():
        run_ragas(ragas_ds, out_dir / "ragas.json")

    # Latency
    queries = [
        "What are the prerequisites of CSE 574?",
        "Who teaches CSE 115?",
        "MS in CSE credit requirements?",
        "What research areas does the department cover?",
    ]
    benchmark_latency(agent, queries, out_dir / "latency.json")

    # Robustness / guardrails
    run_robustness_suite(agent, out_dir / "robustness.json")


if __name__ == "__main__":
    main()
