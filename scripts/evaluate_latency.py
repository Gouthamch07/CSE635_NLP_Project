"""End-to-end latency evaluation against a running backend.

Measures TTFT (time-to-first-token) and total latency by calling the
streaming chat endpoint. Falls back to /chat (non-streaming) if /chat/stream
is unreachable. Pulls per-stage timings from the backend's `latency_trace`
field when present.

Usage (with a running server):
    PYTHONPATH=. python scripts/evaluate_latency.py
    PYTHONPATH=. python scripts/evaluate_latency.py --backend-url http://localhost:8000
    BACKEND_URL=https://ub-cse-chatbot.onrender.com \
        python scripts/evaluate_latency.py
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES = ROOT / "eval" / "test_queries.jsonl"
DEFAULT_OUT_CSV = ROOT / "results" / "latency_results.csv"
DEFAULT_OUT_JSON = ROOT / "results" / "latency_summary.json"
TTFT_TARGET_MS = 2000.0

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("eval.latency")


def load_queries(path: Path) -> list[dict]:
    """Flatten multi-turn entries into individual query rows so we can time each
    user turn independently."""
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "turns" in obj:
                for i, t in enumerate(obj["turns"], 1):
                    rows.append({
                        "id": f"{obj['id']}-t{i}",
                        "query": t["query"],
                        "category": obj.get("category", "follow_up"),
                    })
            else:
                rows.append({
                    "id": obj["id"],
                    "query": obj["query"],
                    "category": obj.get("category", ""),
                })
    return rows


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[idx]


def stream_one(client: httpx.Client, base_url: str, query: str, timeout: float) -> dict:
    """Hit /chat/stream; record TTFT and total. Return raw + backend-reported timings."""
    payload = {"message": query, "user_id": "eval"}
    t_start = time.perf_counter()
    ttft_ms: float | None = None
    total_ms = 0.0
    final: dict = {}
    try:
        with client.stream(
            "POST",
            f"{base_url}/chat/stream",
            json=payload,
            timeout=timeout,
            headers={"Accept": "application/x-ndjson"},
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    event = json.loads(raw_line)
                except Exception:
                    continue
                etype = event.get("type")
                if etype == "token" and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_start) * 1000
                if etype == "done":
                    final = event
                    if ttft_ms is None:
                        # No tokens emitted (e.g. cached / refused) — treat first done as TTFT
                        ttft_ms = (time.perf_counter() - t_start) * 1000
                if etype == "error":
                    raise RuntimeError(event.get("message", "stream error"))
        total_ms = (time.perf_counter() - t_start) * 1000
    except Exception as exc:
        log.warning("stream failed (%s); falling back to /chat", exc)
        return _sync_one(client, base_url, query, timeout)

    lt = final.get("latency_trace", {}) or {}
    return {
        "ttft_ms": round(ttft_ms or total_ms, 1),
        "total_latency_ms": round(total_ms, 1),
        "guardrail_ms": round(float(lt.get("scope_ms", 0.0)), 1),
        "retrieval_ms": round(float(lt.get("tools_ms", 0.0)), 1),
        "rerank_ms": round(_extract_rerank_ms(final), 1),
        "llm_ms": round(float(lt.get("answer_total_ms", 0.0)), 1),
        "success": 1,
        "scope": final.get("scope", ""),
    }


def _sync_one(client: httpx.Client, base_url: str, query: str, timeout: float) -> dict:
    """Fallback: hit /chat (non-streaming) and use backend-reported ttft_ms/total_ms."""
    payload = {"message": query, "user_id": "eval"}
    t_start = time.perf_counter()
    try:
        r = client.post(f"{base_url}/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        body = r.json()
        wall = (time.perf_counter() - t_start) * 1000
        lt = body.get("latency_trace", {}) or {}
        return {
            "ttft_ms": round(float(body.get("ttft_ms", wall)), 1),
            "total_latency_ms": round(float(body.get("total_ms", wall)), 1),
            "guardrail_ms": round(float(lt.get("scope_ms", 0.0)), 1),
            "retrieval_ms": round(float(lt.get("tools_ms", 0.0)), 1),
            "rerank_ms": round(_extract_rerank_ms(body), 1),
            "llm_ms": round(float(lt.get("answer_total_ms", 0.0)), 1),
            "success": 1,
            "scope": body.get("scope", ""),
        }
    except Exception as exc:
        log.warning("sync /chat failed: %s", exc)
        return {
            "ttft_ms": 0.0, "total_latency_ms": 0.0,
            "guardrail_ms": 0.0, "retrieval_ms": 0.0,
            "rerank_ms": 0.0, "llm_ms": 0.0,
            "success": 0, "scope": "",
        }


def _extract_rerank_ms(body: dict) -> float:
    """Pull cross_encoder_ms out of retrieval_trace if available — backend
    HybridRetriever logs it but doesn't surface it as a top-level field."""
    trace = body.get("retrieval_trace") or {}
    # Older shape stores latency in HybridRetriever logs only — no JSON field.
    # Fall back to 0 if not present.
    timings = trace.get("timings") or trace.get("latency") or {}
    if isinstance(timings, dict):
        return float(timings.get("cross_encoder_ms", 0.0))
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument(
        "--backend-url",
        default=os.environ.get("BACKEND_URL", "http://localhost:8000"),
    )
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument(
        "--warmup",
        action="store_true",
        help="Send one throwaway query first (so cold-start doesn't pollute stats)",
    )
    args = ap.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    queries = load_queries(args.queries)
    log.info("loaded %d queries; backend=%s", len(queries), args.backend_url)

    rows = []
    with httpx.Client() as client:
        if args.warmup and queries:
            log.info("warmup ping ...")
            try:
                stream_one(client, args.backend_url, queries[0]["query"], args.timeout)
            except Exception as exc:
                log.warning("warmup failed: %s", exc)

        for i, q in enumerate(queries, 1):
            log.info("[%d/%d] %s", i, len(queries), q["query"][:80])
            timing = stream_one(client, args.backend_url, q["query"], args.timeout)
            rows.append({
                "query_id": q["id"],
                "query": q["query"],
                "category": q.get("category", ""),
                **timing,
            })

    fieldnames = [
        "query_id", "query", "category",
        "ttft_ms", "total_latency_ms",
        "guardrail_ms", "retrieval_ms", "rerank_ms", "llm_ms",
        "success", "scope",
    ]
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    log.info("wrote %s (%d rows)", args.out_csv, len(rows))

    ttfts = [r["ttft_ms"] for r in rows if r.get("success")]
    totals = [r["total_latency_ms"] for r in rows if r.get("success")]
    summary = {
        "num_queries": len(rows),
        "num_successful": len(ttfts),
        "ttft_target_ms": TTFT_TARGET_MS,
        "avg_ttft_ms": round(statistics.mean(ttfts), 1) if ttfts else 0.0,
        "p50_ttft_ms": round(_percentile(ttfts, 0.50), 1),
        "p95_ttft_ms": round(_percentile(ttfts, 0.95), 1),
        "percent_under_2s_ttft": round(
            100.0 * sum(1 for t in ttfts if t < TTFT_TARGET_MS) / max(1, len(ttfts)), 1
        ),
        "avg_total_latency_ms": round(statistics.mean(totals), 1) if totals else 0.0,
        "p50_total_latency_ms": round(_percentile(totals, 0.50), 1),
        "p95_total_latency_ms": round(_percentile(totals, 0.95), 1),
    }
    args.out_json.write_text(json.dumps(summary, indent=2))
    log.info("summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
