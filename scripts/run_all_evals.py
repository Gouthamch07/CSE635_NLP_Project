"""Run every evaluation script in sequence; continue on failure.

Usage:
    PYTHONPATH=. python scripts/run_all_evals.py
    PYTHONPATH=. python scripts/run_all_evals.py --backend-url http://localhost:8000
    PYTHONPATH=. python scripts/run_all_evals.py --skip ragas latency

Exit code is 0 if at least the graph step ran successfully (so the report
artifacts exist even when an individual eval crashes).
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("eval.runall")

# Order matters: ragas writes ragas_raw.jsonl that llm_judge consumes; graphs
# read everyone's outputs.
STEPS: list[tuple[str, str, list[str]]] = [
    # (step_id, script_path, extra_args)
    ("retrieval", "scripts/evaluate_retrieval.py", []),
    ("guardrails", "scripts/evaluate_guardrails.py", []),
    ("latency", "scripts/evaluate_latency.py", ["--warmup"]),
    ("ragas", "scripts/evaluate_ragas.py", []),
    ("llm_judge", "scripts/evaluate_llm_judge.py", []),
    ("graphs", "scripts/make_eval_graphs.py", []),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--backend-url",
        default=os.environ.get("BACKEND_URL", "http://localhost:8000"),
    )
    ap.add_argument(
        "--skip", nargs="*", default=[],
        help="Step IDs to skip (e.g. --skip latency ragas)",
    )
    ap.add_argument(
        "--only", nargs="*", default=[],
        help="Only run these step IDs (e.g. --only retrieval graphs)",
    )
    args = ap.parse_args()

    env = {**os.environ, "BACKEND_URL": args.backend_url, "PYTHONPATH": "."}

    failures: list[str] = []
    successes: list[str] = []
    for step_id, script, extra in STEPS:
        if args.only and step_id not in args.only:
            continue
        if step_id in args.skip:
            log.info("skipping %s (--skip)", step_id)
            continue
        cmd = [sys.executable, script, *extra]
        # Pass --backend-url to scripts that accept it
        if step_id in {"latency", "ragas"}:
            cmd += ["--backend-url", args.backend_url]
        log.info("===== running %s: %s =====", step_id, " ".join(cmd))
        try:
            rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
        except Exception as exc:
            log.warning("%s crashed: %s", step_id, exc)
            failures.append(step_id)
            continue
        if rc != 0:
            log.warning("%s exited with code %d (continuing)", step_id, rc)
            failures.append(step_id)
        else:
            successes.append(step_id)

    log.info("done — successes=%s failures=%s", successes, failures)
    return 0 if "graphs" in successes or not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
