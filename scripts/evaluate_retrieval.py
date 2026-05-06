"""Retrieval evaluation: Hit@k, MRR, retrieval latency.

Loads eval/test_queries.jsonl, calls the in-process HybridRetriever for every
non-out-of-scope query, scores top-k chunks against the expected source
keywords, and writes results/retrieval_eval.csv plus retrieval_summary.json.

Usage:
    PYTHONPATH=. python scripts/evaluate_retrieval.py
    PYTHONPATH=. python scripts/evaluate_retrieval.py --top-k 10 --queries eval/test_queries.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES = ROOT / "eval" / "test_queries.jsonl"
DEFAULT_OUT_CSV = ROOT / "results" / "retrieval_eval.csv"
DEFAULT_OUT_JSON = ROOT / "results" / "retrieval_summary.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("eval.retrieval")


def load_queries(path: Path) -> list[dict]:
    """Load JSONL. Multi-turn entries are flattened — only the LAST turn is
    used for retrieval since the agent answers it with all prior context.
    Out-of-scope entries are skipped (no retrieval expected)."""
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("category") == "out_of_scope":
                continue
            if "turns" in obj:
                last = obj["turns"][-1]
                rows.append({
                    "id": obj["id"],
                    "query": last.get("query", ""),
                    "category": obj.get("category", "follow_up"),
                    "expected_source_keywords": obj.get(
                        "expected_source_keywords", last.get("expected_source_keywords", [])
                    ),
                })
            else:
                rows.append({
                    "id": obj["id"],
                    "query": obj["query"],
                    "category": obj.get("category", ""),
                    "expected_source_keywords": obj.get("expected_source_keywords", []),
                })
    return rows


def keyword_match(text_blob: str, keywords: list[str]) -> bool:
    """Hit if at least 1 keyword appears for small lists (<=2),
    or >=40% of keywords appear for larger lists."""
    if not keywords:
        return False
    blob = (text_blob or "").lower()
    matched = sum(1 for kw in keywords if kw.lower() in blob)
    if len(keywords) <= 2:
        return matched >= 1
    return matched / len(keywords) >= 0.4


def chunk_text_blob(hit: Any) -> str:
    """Concatenate fields we'll match against."""
    if isinstance(hit, dict):
        title = hit.get("title", "") or ""
        url = hit.get("url", "") or ""
        section = hit.get("section", "") or ""
        text = hit.get("text", "") or ""
    else:
        title = getattr(hit, "title", "") or ""
        url = getattr(hit, "url", "") or ""
        section = getattr(hit, "section", "") or ""
        text = getattr(hit, "text", "") or ""
    return " ".join([title, section, url, text])


def evaluate_one(retriever, q: dict, top_k: int) -> dict:
    t0 = time.perf_counter()
    trace = retriever.retrieve(q["query"], final_k=top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    hits = trace.hits[:top_k]

    expected = q.get("expected_source_keywords", []) or []
    rank_of_first = None
    for rank, hit in enumerate(hits, start=1):
        if keyword_match(chunk_text_blob(hit), expected):
            rank_of_first = rank
            break

    hit_at_1 = int(rank_of_first == 1) if rank_of_first else 0
    hit_at_3 = int(rank_of_first is not None and rank_of_first <= 3)
    hit_at_5 = int(rank_of_first is not None and rank_of_first <= 5)
    mrr_q = (1.0 / rank_of_first) if rank_of_first else 0.0

    top1 = hits[0] if hits else None
    top1_title = (top1.title if top1 else "") if hasattr(top1, "title") else (
        top1.get("title", "") if top1 else ""
    )
    top1_url = (top1.url if top1 else "") if hasattr(top1, "url") else (
        top1.get("url", "") if top1 else ""
    )

    return {
        "query_id": q["id"],
        "query": q["query"],
        "category": q.get("category", ""),
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5,
        "rank_of_first_correct": rank_of_first or "",
        "mrr": round(mrr_q, 4),
        "retrieval_ms": round(elapsed_ms, 1),
        "top_1_title": top1_title,
        "top_1_url": top1_url,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    queries = load_queries(args.queries)
    log.info("loaded %d in-domain queries from %s", len(queries), args.queries)

    # Lazy import so module-level errors (e.g. missing torch) don't blow up
    # the whole eval suite — they'd only matter when this script actually runs.
    from config import get_settings
    from ub_cse_bot.rag.hybrid import HybridRetriever

    s = get_settings()
    bm25_path = s.data_dir / "processed" / "bm25.pkl"
    log.info("loading retriever from %s", bm25_path)
    retriever = HybridRetriever.from_disk(bm25_path)
    retriever.warmup()

    rows = []
    for i, q in enumerate(queries, 1):
        log.info("[%d/%d] %s", i, len(queries), q["query"][:80])
        try:
            row = evaluate_one(retriever, q, args.top_k)
        except Exception as exc:
            log.warning("retrieval failed for %s: %s", q["id"], exc)
            row = {
                "query_id": q["id"], "query": q["query"], "category": q.get("category", ""),
                "hit_at_1": 0, "hit_at_3": 0, "hit_at_5": 0,
                "rank_of_first_correct": "", "mrr": 0.0,
                "retrieval_ms": 0.0, "top_1_title": "", "top_1_url": "",
            }
        rows.append(row)

    # CSV
    fieldnames = list(rows[0].keys()) if rows else []
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s (%d rows)", args.out_csv, len(rows))

    # Summary
    n = max(1, len(rows))
    summary = {
        "num_queries": len(rows),
        "hit_at_1": round(sum(r["hit_at_1"] for r in rows) / n, 4),
        "hit_at_3": round(sum(r["hit_at_3"] for r in rows) / n, 4),
        "hit_at_5": round(sum(r["hit_at_5"] for r in rows) / n, 4),
        "mrr": round(sum(r["mrr"] for r in rows) / n, 4),
        "avg_retrieval_ms": round(
            statistics.mean([r["retrieval_ms"] for r in rows]) if rows else 0.0, 1
        ),
        "p50_retrieval_ms": round(
            statistics.median([r["retrieval_ms"] for r in rows]) if rows else 0.0, 1
        ),
        "p95_retrieval_ms": round(
            sorted([r["retrieval_ms"] for r in rows])[int(0.95 * (len(rows) - 1))]
            if rows else 0.0, 1
        ),
    }
    args.out_json.write_text(json.dumps(summary, indent=2))
    log.info("summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
