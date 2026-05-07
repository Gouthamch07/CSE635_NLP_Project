"""RAGAS-style answer quality evaluation.

Pipeline:
  1. Hit the chat backend for every in-domain query
  2. Save the answer + retrieved contexts (sources)
  3. If `ragas` is importable, score faithfulness / answer_relevancy /
     context_precision (and context_recall if a ground_truth field is supplied)
  4. If `ragas` is missing or fails, write a clear note in the summary and a
     CSV with the answers + contexts so evaluate_llm_judge.py can score them

Usage:
    PYTHONPATH=. python scripts/evaluate_ragas.py
    PYTHONPATH=. python scripts/evaluate_ragas.py --backend-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES = ROOT / "eval" / "test_queries.jsonl"
DEFAULT_OUT_CSV = ROOT / "results" / "ragas_eval.csv"
DEFAULT_OUT_JSON = ROOT / "results" / "ragas_summary.json"
DEFAULT_RAW_PATH = ROOT / "results" / "ragas_raw.jsonl"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("eval.ragas")


def load_queries(path: Path) -> list[dict]:
    """Only in-domain non-follow-up queries (RAGAS doesn't model multi-turn well)."""
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
                continue
            rows.append(obj)
    return rows


def fetch_answer(client: httpx.Client, base_url: str, query: str, timeout: float) -> dict:
    """Call /chat (non-streaming) and return {answer, contexts:list[str], sources}."""
    r = client.post(
        f"{base_url}/chat",
        json={"message": query, "user_id": "eval-ragas"},
        timeout=timeout,
    )
    r.raise_for_status()
    body = r.json()
    contexts: list[str] = []

    # Prefer hit text from retrieval_trace if available
    trace = body.get("retrieval_trace") or {}
    for h in (trace.get("hits") or [])[:6]:
        text = h.get("text") if isinstance(h, dict) else None
        if text:
            contexts.append(text)

    # Fallback: pull from sources (title + url + section)
    if not contexts:
        for s in body.get("sources", []) or []:
            blob = " — ".join(filter(None, [
                s.get("title", ""), s.get("section", ""), s.get("url", ""),
            ]))
            if blob:
                contexts.append(blob)

    return {
        "answer": body.get("text", ""),
        "contexts": contexts,
        "sources": body.get("sources", []),
        "scope": body.get("scope", ""),
    }


def try_ragas_score(rows: list[dict]) -> tuple[list[dict], dict | None]:
    """Score with RAGAS if available. Returns (rows_with_scores, summary_or_None)."""
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy
        # Faithfulness: try the modern collections-style implementation first
        # (more lenient JSON parsing on non-OpenAI judges); fall back to the
        # legacy module-level metric.
        try:
            from ragas.metrics.collections import Faithfulness
            faithfulness = Faithfulness()
        except Exception:
            from ragas.metrics import faithfulness  # type: ignore[assignment]
        try:
            from ragas.metrics import context_recall
        except Exception:
            context_recall = None  # type: ignore[assignment]
        # RAGAS 0.2+ split context_precision into two: the default needs a
        # `reference` ground-truth column we don't have; the WithoutReference
        # variant uses only question / answer / contexts (LLM-judged).
        try:
            from ragas.metrics import LLMContextPrecisionWithoutReference
            context_precision_metric = LLMContextPrecisionWithoutReference()
        except Exception:
            try:
                from ragas.metrics import context_precision as context_precision_metric  # legacy
            except Exception:
                context_precision_metric = None
        from datasets import Dataset
    except Exception as exc:
        log.warning("RAGAS not available: %s", exc)
        return rows, None

    # Skip rows with empty answers / contexts (RAGAS will crash on them)
    scorable = [r for r in rows if r["answer"] and r["contexts"]]
    if not scorable:
        log.warning("no scorable rows for RAGAS")
        return rows, None

    dataset_dict = {
        "question": [r["query"] for r in scorable],
        "answer": [r["answer"] for r in scorable],
        "contexts": [r["contexts"] for r in scorable],
    }
    have_gt = all(r.get("ground_truth") for r in scorable)
    if have_gt:
        dataset_dict["ground_truth"] = [r["ground_truth"] for r in scorable]

    ds = Dataset.from_dict(dataset_dict)
    metrics = [faithfulness, answer_relevancy]
    if context_precision_metric is not None:
        metrics.append(context_precision_metric)
    if have_gt and context_recall is not None:
        metrics.append(context_recall)

    # Pick a judge LLM. If OPENAI_API_KEY is set we let RAGAS use its default.
    # Otherwise wire our existing Vertex Gemini as the judge — same creds the
    # chatbot uses, no extra accounts needed.
    judge_llm = None
    judge_embed = None
    if not os.environ.get("OPENAI_API_KEY"):
        try:
            from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
            from config import get_settings
            s = get_settings()
            judge_llm = LangchainLLMWrapper(ChatVertexAI(
                model_name=s.vertex_model,
                project=s.google_cloud_project,
                location=s.google_cloud_location,
                temperature=0,
                # Per-call request timeout — faithfulness / context_precision
                # do claim-by-claim verification and can run long.
                request_timeout=180,
                max_retries=2,
            ))
            judge_embed = LangchainEmbeddingsWrapper(VertexAIEmbeddings(
                model_name=s.vertex_embed_model,
                project=s.google_cloud_project,
                location=s.google_cloud_embed_location or "us-central1",
            ))
            log.info("using Vertex Gemini as RAGAS judge (no OpenAI key set)")
        except Exception as exc:
            log.warning(
                "could not build Vertex judge for RAGAS: %s — install "
                "`langchain-google-vertexai` or set OPENAI_API_KEY",
                exc,
            )
            return rows, None

    # Cap concurrency so we don't pile parallel Gemini calls and trip the
    # global rate limit. Bump per-job timeout to absorb slow faithfulness runs.
    try:
        from ragas.run_config import RunConfig
        run_config = RunConfig(timeout=300, max_workers=4, max_retries=3)
    except Exception:
        run_config = None

    log.info("running ragas.evaluate on %d rows ...", len(scorable))
    # Surface RAGAS internal exceptions so we can see WHY a metric fails (the
    # default RAGAS run swallows per-job exceptions and just emits NaN).
    import logging as _logging
    _logging.getLogger("ragas.executor").setLevel(_logging.DEBUG)
    try:
        result = evaluate(
            ds,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embed,
            run_config=run_config,
            raise_exceptions=False,
        )
    except Exception as exc:
        log.warning("ragas.evaluate failed: %s", exc)
        return rows, None

    df = result.to_pandas()
    by_q = {r["query"]: r for r in scorable}
    # Pull whatever score columns RAGAS produced (names vary between versions).
    score_cols = [c for c in df.columns if c not in {
        "user_input", "question", "response", "answer", "retrieved_contexts",
        "contexts", "reference", "ground_truth",
    }]
    # Friendly canonical names so the summary reads consistently
    rename = {
        "llm_context_precision_without_reference": "context_precision",
        "llm_context_precision_with_reference": "context_precision",
    }
    metric_cols: list[str] = []
    for raw_col in score_cols:
        canon = rename.get(raw_col, raw_col)
        metric_cols.append(canon)
        for _, df_row in df.iterrows():
            q = df_row.get("user_input") or df_row.get("question")
            target = by_q.get(q)
            if target is None:
                continue
            try:
                target[canon] = float(df_row[raw_col]) if df_row[raw_col] is not None else None
            except Exception:
                target[canon] = None

    import math
    summary = {}
    for col in metric_cols:
        values = [
            v for r in scorable
            if (v := r.get(col)) is not None
            and isinstance(v, (int, float))
            and not math.isnan(v)
        ]
        summary[f"avg_{col}"] = round(sum(values) / len(values), 4) if values else None
        summary[f"num_{col}_scored"] = len(values)
    summary["num_scored"] = len(scorable)
    summary["num_total"] = len(rows)
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--raw-jsonl", type=Path, default=DEFAULT_RAW_PATH,
                    help="Where to dump raw answers+contexts so evaluate_llm_judge can read them")
    ap.add_argument(
        "--backend-url",
        default=os.environ.get("BACKEND_URL", "http://localhost:8000"),
    )
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Run only the first N queries (0 = all). Useful for quick smoke runs.",
    )
    args = ap.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    queries = load_queries(args.queries)
    if args.limit > 0:
        queries = queries[: args.limit]
    log.info("loaded %d in-domain queries; backend=%s", len(queries), args.backend_url)

    rows: list[dict] = []
    with httpx.Client() as client:
        for i, q in enumerate(queries, 1):
            log.info("[%d/%d] %s", i, len(queries), q["query"][:80])
            try:
                resp = fetch_answer(client, args.backend_url, q["query"], args.timeout)
            except Exception as exc:
                log.warning("/chat failed for %s: %s", q["id"], exc)
                resp = {"answer": "", "contexts": [], "sources": [], "scope": ""}
            rows.append({
                "query_id": q["id"],
                "query": q["query"],
                "category": q.get("category", ""),
                "answer": resp["answer"],
                "contexts": resp["contexts"],
                "sources": resp["sources"],
                "ground_truth": q.get("ground_truth", ""),
            })

    # Always write raw jsonl (used by evaluate_llm_judge.py)
    with args.raw_jsonl.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log.info("wrote raw answers to %s", args.raw_jsonl)

    rows, ragas_summary = try_ragas_score(rows)

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    fieldnames = ["query_id", "query", "category", "answer", "contexts", *metric_cols]
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "query_id": r["query_id"],
                "query": r["query"],
                "category": r["category"],
                "answer": r["answer"],
                "contexts": " ||| ".join(r.get("contexts") or [])[:8000],
                **{k: r.get(k, "") for k in metric_cols},
            })
    log.info("wrote %s (%d rows)", args.out_csv, len(rows))

    summary = ragas_summary or {
        "ragas_available": False,
        "num_total": len(rows),
        "note": (
            "RAGAS not installed / failed. To enable real RAGAS scoring run:\n"
            "    pip install 'ragas>=0.2' datasets pandas\n"
            "and ensure OPENAI_API_KEY (or another supported judge LLM) is set.\n"
            "Otherwise use scripts/evaluate_llm_judge.py — it scores the same\n"
            "answers (saved to results/ragas_raw.jsonl) with our own Vertex Gemini "
            "judge."
        ),
    }
    if ragas_summary:
        summary = {"ragas_available": True, **ragas_summary}
    args.out_json.write_text(json.dumps(summary, indent=2))
    log.info("summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
