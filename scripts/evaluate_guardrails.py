"""Guardrail (scope classifier) evaluation.

Calls the in-process ScopeClassifier directly when available (fast, no LLM
unless ambiguous). Falls back to the chat backend when --backend-url is set.
Computes accuracy, in-domain accept rate, out-of-scope rejection rate, FP/FN.

Usage:
    PYTHONPATH=. python scripts/evaluate_guardrails.py
    PYTHONPATH=. python scripts/evaluate_guardrails.py --backend-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES = ROOT / "eval" / "test_queries.jsonl"
DEFAULT_OUT_CSV = ROOT / "results" / "guardrail_eval.csv"
DEFAULT_OUT_JSON = ROOT / "results" / "guardrail_summary.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("eval.guardrails")


def load_queries(path: Path) -> list[dict]:
    """Pull every entry; for follow-up multi-turn rows we evaluate only the
    first turn (the disambiguating turn) for guardrail purposes."""
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            expected = obj.get("expected_guardrail_label", "in_domain")
            if "turns" in obj:
                first = obj["turns"][0]
                rows.append({
                    "id": obj["id"],
                    "query": first["query"],
                    "expected_label": expected,
                })
            else:
                rows.append({
                    "id": obj["id"],
                    "query": obj["query"],
                    "expected_label": expected,
                })
    return rows


def _normalize(label: str) -> str:
    """Map ScopeDecision labels and our jsonl labels to the same vocab.
    Treat `small_talk` as in_domain (it gets answered, just briefly)."""
    label = (label or "").strip().lower()
    if label in {"in_scope", "in_domain", "small_talk"}:
        return "in_domain"
    if label == "out_of_scope":
        return "out_of_scope"
    return label or "in_domain"


def predict_in_process(query: str) -> tuple[str, str]:
    """Returns (predicted_label, reason)."""
    from ub_cse_bot.guardrails.scope import ScopeClassifier
    from ub_cse_bot.llm.vertex_client import VertexGemini

    # LLM fallback is allowed but rarely fires — keyword layer covers most cases.
    try:
        clf = ScopeClassifier(llm=VertexGemini())
    except Exception as exc:
        log.warning("VertexGemini unavailable; using keyword-only classifier: %s", exc)
        clf = ScopeClassifier(llm=None)
    decision = clf.classify(query)
    return _normalize(decision.label), decision.reason


def predict_via_backend(query: str, base_url: str, timeout: float) -> tuple[str, str]:
    import httpx

    payload = {"message": query, "user_id": "eval-guard"}
    with httpx.Client(timeout=timeout) as c:
        r = c.post(f"{base_url}/chat", json=payload)
        r.raise_for_status()
        body = r.json()
    return _normalize(body.get("scope", "in_domain")), (body.get("text") or "")[:200]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument(
        "--backend-url",
        default=os.environ.get("BACKEND_URL", ""),
        help="If set, call the running backend instead of the in-process classifier.",
    )
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    queries = load_queries(args.queries)
    log.info("loaded %d queries; mode=%s",
             len(queries), "backend" if args.backend_url else "in-process")

    rows = []
    for i, q in enumerate(queries, 1):
        log.info("[%d/%d] %s", i, len(queries), q["query"][:80])
        try:
            if args.backend_url:
                pred, snippet = predict_via_backend(q["query"], args.backend_url, args.timeout)
            else:
                pred, snippet = predict_in_process(q["query"])
        except Exception as exc:
            log.warning("guardrail prediction failed for %s: %s", q["id"], exc)
            pred, snippet = "in_domain", f"(error: {exc})"

        correct = int(pred == q["expected_label"])
        rows.append({
            "query_id": q["id"],
            "query": q["query"],
            "expected_label": q["expected_label"],
            "predicted_label": pred,
            "correct": correct,
            "response_snippet": (snippet or "")[:200],
        })

    fieldnames = list(rows[0].keys()) if rows else []
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s (%d rows)", args.out_csv, len(rows))

    n = max(1, len(rows))
    in_domain = [r for r in rows if r["expected_label"] == "in_domain"]
    out_of_scope = [r for r in rows if r["expected_label"] == "out_of_scope"]
    # FP = predicted out_of_scope when actually in_domain
    fp = sum(1 for r in in_domain if r["predicted_label"] == "out_of_scope")
    # FN = predicted in_domain when actually out_of_scope (let through bad queries)
    fn = sum(1 for r in out_of_scope if r["predicted_label"] == "in_domain")
    summary = {
        "num_queries": len(rows),
        "accuracy": round(sum(r["correct"] for r in rows) / n, 4),
        "in_domain_accept_rate": round(
            sum(1 for r in in_domain if r["predicted_label"] == "in_domain")
            / max(1, len(in_domain)), 4
        ),
        "out_of_scope_rejection_rate": round(
            sum(1 for r in out_of_scope if r["predicted_label"] == "out_of_scope")
            / max(1, len(out_of_scope)), 4
        ),
        "false_positives": fp,
        "false_negatives": fn,
        "num_in_domain": len(in_domain),
        "num_out_of_scope": len(out_of_scope),
    }
    args.out_json.write_text(json.dumps(summary, indent=2))
    log.info("summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
