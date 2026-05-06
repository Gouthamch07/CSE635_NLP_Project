"""LLM-as-judge fallback for answer quality.

Reads results/ragas_raw.jsonl (produced by evaluate_ragas.py) and asks the
project's existing Vertex Gemini client to score each (query, answer, contexts)
triple on:
  - groundedness: 1-5 (does the answer follow from the contexts?)
  - relevance: 1-5 (does it answer the question asked?)
  - completeness: 1-5 (does it cover the relevant facets?)
  - task_success: bool (would a UB CSE student be satisfied?)

Writes results/task_success_eval.csv + task_success_summary.json.
Use this when RAGAS isn't installed/working.

Usage:
    PYTHONPATH=. python scripts/evaluate_llm_judge.py
    PYTHONPATH=. python scripts/evaluate_llm_judge.py --raw results/ragas_raw.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = ROOT / "results" / "ragas_raw.jsonl"
DEFAULT_OUT_CSV = ROOT / "results" / "task_success_eval.csv"
DEFAULT_OUT_JSON = ROOT / "results" / "task_success_summary.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("eval.judge")

JUDGE_SYSTEM = (
    "You are a strict evaluator of a chatbot for the University at Buffalo "
    "Computer Science and Engineering (UB CSE) department. You grade ONE "
    "(question, answer, retrieved_contexts) tuple at a time and return ONLY "
    "a JSON object with these exact keys:\n"
    '  "groundedness": int 1-5 (does every factual claim in the answer follow '
    'from the retrieved contexts? 5 = perfectly grounded, 1 = hallucinated)\n'
    '  "relevance": int 1-5 (does the answer address the question?)\n'
    '  "completeness": int 1-5 (does it cover the obvious facets the user '
    'would expect?)\n'
    '  "task_success": boolean (would a UB CSE student be satisfied?)\n'
    '  "rationale": short string (<=150 chars)\n'
    "Return only the JSON object, nothing else."
)


def build_judge_prompt(query: str, answer: str, contexts: list[str]) -> str:
    ctx_blob = "\n---\n".join(contexts[:6])[:6000] if contexts else "(no contexts)"
    return (
        f"QUESTION:\n{query}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"RETRIEVED CONTEXTS:\n{ctx_blob}\n\n"
        "Now grade and return JSON only."
    )


def parse_judge_json(raw: str) -> dict | None:
    """LLM may wrap JSON in ```json ... ``` — strip and find the first balanced object."""
    if not raw:
        return None
    s = raw.strip()
    # strip markdown fence
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.rsplit("```", 1)[0]
    # find first { ... }
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(s[start : end + 1])
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not args.raw.exists():
        log.error("raw answers file not found: %s", args.raw)
        log.error("run scripts/evaluate_ragas.py first to generate it")
        return 2

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in args.raw.read_text().splitlines() if line.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]
    log.info("loaded %d rows to judge from %s", len(rows), args.raw)

    from ub_cse_bot.llm.vertex_client import LLMMessage, VertexGemini

    judge = VertexGemini()

    out_rows = []
    for i, r in enumerate(rows, 1):
        log.info("[%d/%d] %s", i, len(rows), (r.get("query") or "")[:80])
        if not r.get("answer"):
            out_rows.append({
                "query_id": r["query_id"], "query": r["query"],
                "groundedness": 0, "relevance": 0, "completeness": 0,
                "task_success": False, "rationale": "no answer to grade",
            })
            continue

        prompt = build_judge_prompt(r["query"], r["answer"], r.get("contexts") or [])
        try:
            raw = judge.generate(
                [LLMMessage("system", JUDGE_SYSTEM), LLMMessage("user", prompt)],
                temperature=0.0,
                max_output_tokens=512,
            )
        except Exception as exc:
            log.warning("judge LLM failed: %s", exc)
            raw = ""

        parsed = parse_judge_json(raw) or {}
        out_rows.append({
            "query_id": r["query_id"],
            "query": r["query"],
            "groundedness": int(parsed.get("groundedness", 0) or 0),
            "relevance": int(parsed.get("relevance", 0) or 0),
            "completeness": int(parsed.get("completeness", 0) or 0),
            "task_success": bool(parsed.get("task_success", False)),
            "rationale": (parsed.get("rationale") or "")[:200],
        })

    fieldnames = ["query_id", "query", "groundedness", "relevance",
                  "completeness", "task_success", "rationale"]
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    log.info("wrote %s (%d rows)", args.out_csv, len(out_rows))

    def _avg(field: str) -> float:
        vals = [r[field] for r in out_rows if isinstance(r.get(field), (int, float)) and r[field] > 0]
        return round(statistics.mean(vals), 3) if vals else 0.0

    n = max(1, len(out_rows))
    summary = {
        "num_judged": len(out_rows),
        "avg_groundedness": _avg("groundedness"),
        "avg_relevance": _avg("relevance"),
        "avg_completeness": _avg("completeness"),
        "task_success_rate": round(
            sum(1 for r in out_rows if r["task_success"]) / n, 4
        ),
    }
    args.out_json.write_text(json.dumps(summary, indent=2))
    log.info("summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
