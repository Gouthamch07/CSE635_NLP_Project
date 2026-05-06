# Evaluation pipeline (CSE 635 final report)

End-to-end metrics for the UB CSE Assistant: retrieval quality, latency,
guardrails, and answer quality. Every script writes a CSV (per-query) plus a
JSON summary that the graph generator and the report table consume.

## Quick start

### 1. Start the backend

```bash
source .venv/bin/activate
export PYTHONPATH=.
# (Pinecone / Vertex / Neo4j env from RUNBOOK §2)
export VERTEX_THINKING_LEVEL=MINIMAL
python3 -m uvicorn ub_cse_bot.ui.server:app --port 8000
```

Wait until you see `Application startup complete.`

### 2. Run all evaluations in one shot

```bash
PYTHONPATH=. python scripts/run_all_evals.py
```

Or against the deployed Render instance:

```bash
PYTHONPATH=. python scripts/run_all_evals.py \
    --backend-url https://ub-cse-chatbot.onrender.com
```

If a single step fails the runner logs a warning and continues. To re-run a
subset:

```bash
PYTHONPATH=. python scripts/run_all_evals.py --only retrieval graphs
PYTHONPATH=. python scripts/run_all_evals.py --skip ragas
```

## Individual scripts

| Script | Needs server? | Output |
|---|---|---|
| `scripts/evaluate_retrieval.py` | No (uses HybridRetriever in-process) | `results/retrieval_eval.csv`, `results/retrieval_summary.json` |
| `scripts/evaluate_latency.py` | **Yes** | `results/latency_results.csv`, `results/latency_summary.json` |
| `scripts/evaluate_guardrails.py` | No by default; pass `--backend-url` to use server | `results/guardrail_eval.csv`, `results/guardrail_summary.json` |
| `scripts/evaluate_ragas.py` | **Yes** | `results/ragas_eval.csv`, `results/ragas_summary.json`, `results/ragas_raw.jsonl` |
| `scripts/evaluate_llm_judge.py` | No (reads `ragas_raw.jsonl`; calls Vertex Gemini directly) | `results/task_success_eval.csv`, `results/task_success_summary.json` |
| `scripts/make_eval_graphs.py` | No | `results/graphs/*.png` |

All scripts use `argparse`. Common flags:

```bash
python scripts/evaluate_retrieval.py --top-k 10
python scripts/evaluate_latency.py --backend-url https://… --timeout 180 --warmup
python scripts/evaluate_ragas.py --limit 5     # quick smoke run
python scripts/evaluate_guardrails.py --backend-url http://localhost:8000
```

The backend URL also reads from the `BACKEND_URL` environment variable when no
`--backend-url` flag is given.

## Test set (`eval/test_queries.jsonl`)

50 entries across these categories:

| Category | Count | Purpose |
|---|---:|---|
| `program_requirements` | 11 | MS / BS / PhD / specializations / capstone |
| `course` | 10 | CSE 574, 521, 531, 565, 546, 535, 250, 562, 589 + one extra |
| `faculty`, `research` | 5 | "who teaches X", labs by area |
| `policies`, `general`, `admissions` | 4 | academic integrity, contact, scholarships, deadlines |
| `follow_up` | 5 | multi-turn (`turns:[…]`) |
| `out_of_scope` | 10 | pizza, weather, NFL, etc. |

Each row carries `expected_source_keywords` and `expected_answer_keywords` so
retrieval and guardrails can be scored without a human in the loop.

## What each metric means

- **Hit@k** — fraction of in-domain queries where one of the top *k* retrieved
  chunks contains enough `expected_source_keywords` (≥1 for tiny lists, ≥40%
  otherwise).
- **MRR** — mean reciprocal rank of the first correct chunk across queries; 1.0
  is perfect, 0 means never found.
- **TTFT** — time from sending the request to receiving the first streamed
  token. Hard target for this project: **< 2000 ms**.
- **% TTFT < 2 s** — share of successful runs hitting the target.
- **Total latency** — request start → final answer chunk.
- **RAGAS faithfulness** — every claim in the answer is supported by retrieved
  context.
- **RAGAS answer relevancy** — answer addresses the question asked.
- **RAGAS context precision** — retrieved contexts are mostly relevant
  (less noise = higher).
- **Guardrail accuracy** — % of queries where the scope classifier's label
  matches the expected `in_domain` / `out_of_scope` label.
- **Task success rate** — LLM judge vote that "a UB CSE student would be
  satisfied". Useful when RAGAS isn't installed.

## Final-report table template

Fill in from `results/*_summary.json` after the run:

| Metric | Result |
|---|---:|
| Avg TTFT | X.XX s |
| P50 TTFT | X.XX s |
| P95 TTFT | X.XX s |
| % TTFT < 2 s | XX.X % |
| Avg full latency | X.XX s |
| Hit@1 | XX.X % |
| Hit@3 | XX.X % |
| Hit@5 | XX.X % |
| MRR | 0.XX |
| RAGAS Faithfulness | 0.XX |
| Answer Relevance | 0.XX |
| Context Precision | 0.XX |
| Guardrail Accuracy | XX.X % |
| Task Success Rate | XX.X % |

## Graphs to include in the ACL report

The most informative figures (in `results/graphs/`):

1. **`latency_ttft_distribution.png`** — histogram with the 2 s target line.
   The single most important figure for the latency story.
2. **`retrieval_hit_rate.png`** — Hit@1/3/5 + MRR bars. Shows the hybrid
   retriever's effectiveness.
3. **`ragas_scores.png`** — faithfulness, answer relevancy, context precision.
   Falls back to LLM-judge groundedness/relevance/completeness if RAGAS isn't
   installed.
4. **`guardrail_accuracy.png`** — in-domain accept vs. out-of-scope reject.
   Shows the safety story.

The remaining two (`latency_summary.png`, `task_success.png`) are useful but
optional.

## Common gotchas

- **`ModuleNotFoundError: ub_cse_bot`** — set `PYTHONPATH=.` (the run-all
  script does this automatically).
- **`bm25.pkl` missing** — required by `evaluate_retrieval.py`. It's checked
  into git at `data/processed/bm25.pkl`. If you removed it, regenerate with
  `python scripts/rebuild_bm25_from_pinecone.py`.
- **Latency eval gets a 502 / hangs** — the server isn't running on the
  expected URL. Test with `curl http://localhost:8000/`.
- **RAGAS install errors** — RAGAS pulls heavy ML deps and needs an OpenAI key
  by default. If install fails, just use `evaluate_llm_judge.py` instead — it
  scores the same answers using the project's existing Vertex Gemini.
- **First chat is slow** — cross-encoder model + Vertex warmup. The latency
  script's `--warmup` flag (set by default in `run_all_evals.py`) absorbs that
  cold-start so it doesn't pollute stats.

## Re-running after code changes

The eval set is deterministic so all numbers are directly comparable across
runs. Suggested commits to keep:

```
git add results/ -A
git commit -m "Eval results: <commit-id> <model> <date>"
```

The `eval/` folder, `scripts/evaluate_*.py`, and `results/` are tracked; raw
caches and intermediate `__pycache__` are gitignored.
