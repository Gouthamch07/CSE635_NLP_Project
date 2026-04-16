from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Iterable

from ..agent import UBCSEAgent
from ..utils.io import read_jsonl, write_json
from ..utils.logging import get_logger

log = get_logger(__name__)


def benchmark_latency(agent: UBCSEAgent, queries: Iterable[str], out: Path) -> dict:
    """Measure total latency and approximate TTFT via the streaming API."""
    rows: list[dict] = []
    for q in queries:
        t0 = time.time()
        first = None
        text_parts: list[str] = []
        for tok in agent.stream(q):
            if first is None:
                first = time.time()
            text_parts.append(tok)
        total = time.time() - t0
        ttft = (first - t0) if first else total
        rows.append({
            "query": q, "ttft_s": round(ttft, 3),
            "total_s": round(total, 3),
            "chars": sum(len(p) for p in text_parts),
        })
    ttfts = [r["ttft_s"] for r in rows]
    totals = [r["total_s"] for r in rows]
    summary = {
        "ttft_p50": statistics.median(ttfts) if ttfts else 0,
        "ttft_p90": statistics.quantiles(ttfts, n=10)[-1] if len(ttfts) >= 10 else max(ttfts, default=0),
        "total_p50": statistics.median(totals) if totals else 0,
        "under_2s_ttft_pct": sum(1 for t in ttfts if t < 2.0) / max(1, len(ttfts)),
        "n": len(rows),
    }
    write_json(out, {"summary": summary, "rows": rows})
    log.info("latency %s", summary)
    return summary
