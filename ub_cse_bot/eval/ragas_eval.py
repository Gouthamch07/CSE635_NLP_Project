from __future__ import annotations

from pathlib import Path

from ..utils.io import read_jsonl, write_json
from ..utils.logging import get_logger

log = get_logger(__name__)


def run_ragas(dataset_path: Path, out_path: Path) -> dict:
    """Run RAGAS faithfulness + answer_relevancy + context_precision.

    Dataset schema (jsonl):
        {"question": "...", "answer": "...", "contexts": ["...", "..."], "ground_truth": "..."}
    """
    rows = read_jsonl(dataset_path)
    if not rows:
        raise ValueError(f"empty dataset: {dataset_path}")

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    ds = Dataset.from_list(rows)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    scores = {k: float(v) for k, v in result.to_pandas().mean(numeric_only=True).to_dict().items()}
    write_json(out_path, scores)
    log.info("RAGAS scores -> %s", scores)
    return scores
