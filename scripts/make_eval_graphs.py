"""Generate report-ready PNG graphs from results/*.csv and *.json.

Reads whatever exists in results/ — if a particular eval hasn't been run,
that graph is skipped (with a warning) instead of crashing the whole script.

Usage:
    python scripts/make_eval_graphs.py
    python scripts/make_eval_graphs.py --results-dir custom/results --graphs-dir custom/graphs
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_GRAPHS = ROOT / "results" / "graphs"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("eval.graphs")


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("wrote %s", path)


def latency_ttft_distribution(results_dir: Path, graphs_dir: Path) -> None:
    rows = _load_csv(results_dir / "latency_results.csv")
    if not rows:
        log.warning("skipping ttft_distribution — latency_results.csv missing")
        return
    ttfts = [float(r["ttft_ms"]) for r in rows if r.get("ttft_ms") and r.get("success") == "1"]
    if not ttfts:
        log.warning("skipping ttft_distribution — no successful runs")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ttfts, bins=15, edgecolor="black", color="#0072CE")
    ax.axvline(2000, color="#C8102E", linestyle="--", linewidth=2, label="2 s target")
    ax.set_title("TTFT distribution (lower = better)")
    ax.set_xlabel("Time-to-first-token (ms)")
    ax.set_ylabel("Number of queries")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, graphs_dir / "latency_ttft_distribution.png")


def latency_summary_bars(results_dir: Path, graphs_dir: Path) -> None:
    summary = _load_json(results_dir / "latency_summary.json")
    if not summary:
        log.warning("skipping latency_summary — latency_summary.json missing")
        return
    labels = ["P50 TTFT", "P95 TTFT", "Avg total"]
    values = [
        summary.get("p50_ttft_ms", 0) / 1000,
        summary.get("p95_ttft_ms", 0) / 1000,
        summary.get("avg_total_latency_ms", 0) / 1000,
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=["#0072CE", "#005BBB", "#003A70"])
    ax.axhline(2.0, color="#C8102E", linestyle="--", linewidth=2, label="2 s target")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f} s",
                ha="center", va="bottom")
    ax.set_title("Latency summary")
    ax.set_ylabel("Seconds")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, graphs_dir / "latency_summary.png")


def retrieval_hit_rate(results_dir: Path, graphs_dir: Path) -> None:
    summary = _load_json(results_dir / "retrieval_summary.json")
    if not summary:
        log.warning("skipping retrieval_hit_rate — retrieval_summary.json missing")
        return
    labels = ["Hit@1", "Hit@3", "Hit@5", "MRR"]
    values = [
        summary.get("hit_at_1", 0),
        summary.get("hit_at_3", 0),
        summary.get("hit_at_5", 0),
        summary.get("mrr", 0),
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=["#0072CE", "#005BBB", "#003A70", "#18A489"])
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}",
                ha="center", va="bottom")
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval Hit@k and MRR")
    ax.set_ylabel("Score (0–1)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, graphs_dir / "retrieval_hit_rate.png")


def ragas_scores(results_dir: Path, graphs_dir: Path) -> None:
    summary = _load_json(results_dir / "ragas_summary.json")
    if not summary or not summary.get("ragas_available"):
        # Fall back to LLM-judge scores
        summary = _load_json(results_dir / "task_success_summary.json")
        if not summary:
            log.warning("skipping ragas_scores — neither ragas_summary nor task_success_summary present")
            return
        labels = ["Groundedness", "Relevance", "Completeness"]
        values = [
            summary.get("avg_groundedness", 0) / 5.0,
            summary.get("avg_relevance", 0) / 5.0,
            summary.get("avg_completeness", 0) / 5.0,
        ]
        title = "Answer quality (LLM judge, normalized 0–1)"
    else:
        labels = ["Faithfulness", "Answer Relevancy", "Context Precision"]
        values = [
            summary.get("avg_faithfulness", 0) or 0,
            summary.get("avg_answer_relevancy", 0) or 0,
            summary.get("avg_context_precision", 0) or 0,
        ]
        if summary.get("avg_context_recall") is not None:
            labels.append("Context Recall")
            values.append(summary.get("avg_context_recall", 0) or 0)
        title = "RAGAS scores"

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#0072CE", "#18A489", "#005BBB", "#003A70"][: len(labels)])
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}",
                ha="center", va="bottom")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.set_ylabel("Score (0–1)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, graphs_dir / "ragas_scores.png")


def guardrail_accuracy(results_dir: Path, graphs_dir: Path) -> None:
    summary = _load_json(results_dir / "guardrail_summary.json")
    if not summary:
        log.warning("skipping guardrail_accuracy — guardrail_summary.json missing")
        return
    labels = ["In-domain accept", "Out-of-scope reject", "Overall accuracy"]
    values = [
        summary.get("in_domain_accept_rate", 0),
        summary.get("out_of_scope_rejection_rate", 0),
        summary.get("accuracy", 0),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#18A489", "#C8102E", "#0072CE"])
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{100*val:.1f}%",
                ha="center", va="bottom")
    ax.set_ylim(0, 1.05)
    ax.set_title("Guardrail accuracy")
    ax.set_ylabel("Rate (0–1)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, graphs_dir / "guardrail_accuracy.png")


def task_success(results_dir: Path, graphs_dir: Path) -> None:
    summary = _load_json(results_dir / "task_success_summary.json")
    if not summary:
        log.warning("skipping task_success — task_success_summary.json missing")
        return
    rate = summary.get("task_success_rate", 0)
    fail_rate = max(0.0, 1.0 - rate)
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Success", "Failure"], [rate, fail_rate],
                  color=["#18A489", "#C8102E"])
    for bar, val in zip(bars, [rate, fail_rate]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{100*val:.1f}%",
                ha="center", va="bottom")
    ax.set_ylim(0, 1.05)
    ax.set_title("Task success rate (LLM judge)")
    ax.set_ylabel("Rate (0–1)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, graphs_dir / "task_success.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--graphs-dir", type=Path, default=DEFAULT_GRAPHS)
    args = ap.parse_args()

    args.graphs_dir.mkdir(parents=True, exist_ok=True)
    log.info("results=%s graphs=%s", args.results_dir, args.graphs_dir)

    plotters = [
        latency_ttft_distribution,
        latency_summary_bars,
        retrieval_hit_rate,
        ragas_scores,
        guardrail_accuracy,
        task_success,
    ]
    for fn in plotters:
        try:
            fn(args.results_dir, args.graphs_dir)
        except Exception as exc:
            log.warning("%s failed: %s", fn.__name__, exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
