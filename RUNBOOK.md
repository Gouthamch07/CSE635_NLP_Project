# UB CSE Chatbot — Runbook

Operational guide for running, maintaining, and extending the chatbot. Pairs with [README.md](README.md) which covers the architectural overview.

---

## 1 · Current state

| Component | Status | Notes |
|---|---|---|
| Pinecone (dense, 768-d Vertex `text-embedding-004`) | ✅ 3,230 records, namespace `default` | UB CSE web pages + selected `cse.buffalo.edu/~prof/` syllabi |
| BM25 sparse (local pickle) | ✅ `data/processed/bm25.pkl` | Refit on every `add_url.py` call |
| Neo4j Aura (KG) | ⚠️ Offline | `kg_store=None` is passed to the agent; KG tools degrade gracefully ([tools.py:61](ub_cse_bot/agent/tools.py:61)). Bonus features ("course → related labs") return empty. |
| Vertex Gemini 2.5 Flash-Lite (LLM) | ✅ via ADC | `thinking_budget=0` to fully disable thinking → lowest TTFT in the Gemini family. Quality is sufficient for RAG-grounded factual answers |
| Cross-encoder reranker | ✅ sentence-transformers fallback | `BAAI/bge-reranker-v2-m3` if `FlagEmbedding` is installed |
| Custom FastAPI + HTML UI | ✅ `ub_cse_bot/ui/server.py` | UB-branded; runs on port 8000 |
| Streamlit UI (legacy) | ✅ `ub_cse_bot/ui/app.py` | Still works, kept as fallback |

---

## 2 · Quick start (launch the chatbot)

```bash
cd /Users/mukeshreddypochamreddy/Downloads/CSE635_NLP_Project-main
source .venv/bin/activate

# env (export each new shell, or add to ~/.zshrc)
export PINECONE_API_KEY="pcsk_..."
export PINECONE_INDEX="ub-cse-chatbot"
export PINECONE_REGION="us-east-1"
export GOOGLE_CLOUD_PROJECT="uplifted-valor-474623-c9"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_CLOUD_EMBED_LOCATION="us-central1"
export WARM_START_ON_STARTUP=true
export ENABLE_LLM_PLANNER=false
export CROSS_ENCODER_CANDIDATE_K=12
export ENABLE_LATENCY_DEBUG=true
export VERTEX_MODEL=gemini-3-flash-preview
export VERTEX_THINKING_BUDGET=0
export ANSWER_CONTEXT_K=6
export ANSWER_CONTEXT_CHARS=1400
export ANSWER_MAX_OUTPUT_TOKENS=1600
export CONCISE_ANSWERS=false
# GOOGLE_APPLICATION_CREDENTIALS NOT set — using ADC

# launch
PYTHONPATH=. python3 -m uvicorn ub_cse_bot.ui.server:app --port 8000
```

Open http://localhost:8000.

For auto-reload during development:
```bash
PYTHONPATH=. python3 -m uvicorn ub_cse_bot.ui.server:app --port 8000 --reload
```

---

## 3 · One-time setup (already done — listed for reference / reinstall)

### 3.1 Python venv + minimum deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pinecone rank-bm25 pydantic-settings python-dotenv \
            streamlit fastapi 'uvicorn[standard]' \
            google-cloud-aiplatform vertexai \
            sentence-transformers httpx beautifulsoup4
```

> The full `requirements.txt` includes crawl4ai + playwright + FlagEmbedding which are **not needed at runtime** — only for re-crawling and an optional better reranker. Skip them unless you specifically need to re-crawl.

### 3.2 Vertex authentication (Application Default Credentials)

```bash
brew install --cask google-cloud-sdk
gcloud auth application-default login
gcloud auth application-default set-quota-project uplifted-valor-474623-c9
```

Verify:
```bash
PYTHONPATH=. python3 -c "import vertexai; vertexai.init(project='uplifted-valor-474623-c9', location='us-central1'); from vertexai.language_models import TextEmbeddingModel; m=TextEmbeddingModel.from_pretrained('text-embedding-004'); print('OK', len(m.get_embeddings(['hello'])[0].values))"
```
Should print `OK 768`.

### 3.3 Rebuild BM25 from Pinecone (if `bm25.pkl` is missing)

If you have Pinecone populated but `data/processed/bm25.pkl` is missing:
```bash
PYTHONPATH=. python3 scripts/rebuild_bm25_from_pinecone.py
```
Pulls all chunk text from Pinecone metadata, re-fits BM25, saves pickle. Takes ~1 minute. No re-crawl needed.

---

## 4 · Architecture (one-screen summary)

```
                ┌──────────────┐
   user query →│  Guardrail   │── out_of_scope → friendly redirect (Yelp/weather/etc.)
                └──────┬───────┘── small_talk    → canned reply (no retrieval)
                       │
                       ▼ in_scope
                ┌──────────────┐
                │ Plan (Gemini)│── decides which tools to call
                └──────┬───────┘
                       │
        ┌──────────────┼───────────────┐
        ▼              ▼               ▼
   ┌────────┐    ┌──────────┐    ┌──────────┐
   │ BM25   │    │ Pinecone │    │ Neo4j    │  ← currently disabled
   │ sparse │    │ dense    │    │ KG tools │
   └───┬────┘    └────┬─────┘    └────┬─────┘
       │              │               │
       └──────► RRF fusion ◄──────────┘
                      │
                      ▼
            Lexical rerank (token-overlap boost for course codes)
                      │
                      ▼
            Cross-encoder rerank (BGE bge-reranker-v2-m3)
                      │
                      ▼
              Top 8 chunks → final Gemini call → answer + cited sources
```

Every stage's hits + scores are surfaced in the **Retrieval log** drawer in the UI.

---

## 5 · Adding new data

### 5.1 Add a single URL (fastest)
```bash
PYTHONPATH=. python3 scripts/add_url.py "https://cse.buffalo.edu/~prof/teaching/.../syllabus.html"
```
Fetches → chunks → embeds with Vertex → upserts to Pinecone → refits BM25. **Restart uvicorn** afterwards so the new pickle is loaded into memory.

### 5.2 Full re-crawl (heavy — only if needed)
Requires the full `requirements.txt`:
```bash
pip install -r requirements.txt
playwright install chromium
python scripts/run_crawl.py     # → data/raw/corpus.jsonl
python scripts/build_index.py   # → BM25 + Pinecone (overwrites)
```
To include `cse.buffalo.edu/~prof/` pages, edit `.env` (or export):
```
CRAWL_ALLOWED_DOMAINS=engineering.buffalo.edu,buffalo.edu,cse.buffalo.edu
CRAWL_MAX_DEPTH=4
```

### 5.3 Build the knowledge graph
Requires Neo4j Aura credentials and a corpus.jsonl:
```bash
python scripts/build_kg.py     # → loads Course/Faculty/Lab/Program nodes + relationships
```
Then export `NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD`, restart uvicorn, and update `ui/server.py` to construct a `Neo4jStore` and pass it as `kg_store=` to the agent.

---

## 6 · Evaluation

```bash
cp data/eval/retrieval.jsonl.example data/eval/retrieval.jsonl
cp data/eval/ragas.jsonl.example     data/eval/ragas.jsonl
pip install ragas datasets
PYTHONPATH=. python3 scripts/run_eval.py
# → data/eval_out/{hit_rate.json, ragas.json, latency.json, robustness.json}
```

Outputs:
- **`hit_rate.json`** — Hit@K and MRR over `data/eval/retrieval.jsonl`
- **`ragas.json`** — faithfulness / context-precision / answer-relevancy per question
- **`latency.json`** — TTFT distribution (rubric requires < 2s)
- **`robustness.json`** — out-of-scope refusal rate

Add more questions to the `.jsonl` files for richer eval — one JSON object per line with the schema in the `.example` files.

---

## 7 · UI features (what to demo)

| Feature | Where | Rubric tie-in |
|---|---|---|
| Welcome screen with 4 prompt cards | Initial load | UX polish |
| Markdown rendering in answers | Bot bubbles | Polish |
| Numbered source chips with deep links | Below each answer | Grounding / faithfulness |
| Scope pill (green / red / purple) | Below each answer | Guardrail demonstration |
| TTFT + total ms pills | Below each answer | Latency benchmarking |
| **Retrieval log drawer** | Top-right `Retrieval log` button | Bonus — shows BM25 / Dense / RRF / Lexical / Cross-encoder per stage with scores |
| New chat button | Top-right `+` | Clears server-side `ConversationMemory` |
| Light/dark theme toggle | Top-right moon/sun | Polish |
| Persistent `user_id` (localStorage) | Auto | Personalized memory survives reloads |
| Friendly out-of-scope redirects | Pizza/weather/sports/etc. | Guardrail with topic-aware suggestions |

---

## 8 · Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: config` running a script | Project root not on `sys.path` | Prefix with `PYTHONPATH=.` |
| `ModuleNotFoundError: pydantic_settings` | Wrong Python — using brew's, not the venv | `source .venv/bin/activate` and use `python3 -m uvicorn ...` |
| `pinecone-client` is renamed error | Package moved to `pinecone` | `pip uninstall -y pinecone-client && pip install pinecone` |
| `invalid_grant: Bad Request` from Vertex | ADC refresh token expired (>7 days) | `gcloud auth application-default login` |
| `aiplatform.googleapis.com requires a quota project` | ADC has no quota project | `gcloud auth application-default set-quota-project <PROJECT_ID>` |
| Bot answer cut off mid-sentence | Pro's mandatory thinking ate `max_output_tokens` | Switched to Gemini 2.5 Flash with `thinking_budget=0` + 4096 tokens in [vertex_client.py](ub_cse_bot/llm/vertex_client.py) |
| Trace drawer shows "0 hits" everywhere | Frontend reading wrong field | Already fixed in [app.js](ub_cse_bot/ui/static/app.js) — reads `stage.scores`, joins with `trace.hits` for titles |
| Wall of `torchvision` warnings on streamlit | Streamlit's file watcher walks every transformers submodule | Use `--server.fileWatcherType=none` or just ignore (cosmetic) |
| `cse.buffalo.edu/~prof/...` pages not in answers | Different subdomain than the original crawl scope | Use `scripts/add_url.py` per page, or re-crawl with `cse.buffalo.edu` in `CRAWL_ALLOWED_DOMAINS` |
| `ModuleNotFoundError: orjson` running `rebuild_bm25_*.py` | Triggers full `ub_cse_bot.rag` package init | The script is now self-contained — pull a fresh copy |
| KG queries ("who teaches X", "labs related to Y") return empty | Neo4j isn't connected | Reload KG into a fresh Aura instance and pass `kg_store=` to the agent in `ui/server.py` |

---

## 9 · Key files

```
ub_cse_bot/
├── agent/orchestrator.py      ← plan + tool-call + answer flow
├── agent/tools.py             ← retrieve / KG tools (KG paths gated on kg_store)
├── guardrails/scope.py        ← keyword + LLM scope classifier; friendly redirects
├── rag/hybrid.py              ← HybridRetriever (BM25 + Pinecone + RRF + reranks)
├── rag/bm25.py                ← BM25Index (load/save pickle)
├── rag/pinecone_store.py      ← Pinecone wrapper
├── rag/reranker.py            ← LexicalReranker + CrossEncoderReranker (BGE / ST fallback)
├── llm/vertex_client.py       ← Gemini wrapper (Flash, thinking_budget=0, 4096 tokens)
├── embeddings/lazy.py         ← Disk-cached embedder
├── embeddings/contextual.py   ← Heading-aware contextual chunker
├── ui/server.py               ← FastAPI app for the chatbot
├── ui/app.py                  ← Streamlit fallback
└── ui/static/                 ← index.html / style.css / app.js (UB-themed)

scripts/
├── run_crawl.py
├── build_index.py
├── build_kg.py
├── run_eval.py
├── rebuild_bm25_from_pinecone.py   ← rebuild BM25 if pickle is lost
└── add_url.py                       ← single-page ingest (Pinecone + BM25)
```

---

## 10 · Known gaps (vs. project rubric)

| Rubric requirement | Status | Gap |
|---|---|---|
| Recursive web crawl + PDF extraction | ✅ done (3,230 chunks) | Subdomain `cse.buffalo.edu/~prof/` not fully covered — fixable per-URL with `add_url.py` |
| Knowledge graph + relational queries | ⚠️ code present, instance offline | Need to clone old Aura instance OR rebuild KG into the new one |
| **BONUS** course → related labs/faculty | ⚠️ depends on KG | Same as above |
| Hybrid retrieval (semantic + keyword) | ✅ done | RRF fusion of BM25 + Pinecone |
| **BONUS** lexical + cross-encoder rerank with UI logs | ✅ done | Trace drawer shows all stages |
| Out-of-scope guardrail | ✅ done | Two-stage (keyword + LLM) with friendly topic-aware redirects |
| Dialogue state / follow-ups | ✅ done | `ConversationMemory(max_turns=12)` |
| **BONUS** personalized memory toggle | ✅ done | `PersonalMemory` + `enabled_for(user_id)` |
| RAGAS faithfulness / Hit@K / latency / robustness | ⚠️ harness exists, not yet run | `pip install ragas datasets && python scripts/run_eval.py` |
| TTFT < 2 s | ⚠️ needs fresh measurement | Warm-start is enabled at app boot; demo path skips the extra planner LLM call by default. Use a Flash model if Pro is still too slow. |

Latency switches:
- `WARM_START_ON_STARTUP=true` primes Vertex, Pinecone, embeddings, and cross-encoder before the first user query.
- `ENABLE_LLM_PLANNER=false` uses the deterministic retrieve-first route, saving one Gemini call per normal CSE question.
- `CROSS_ENCODER_CANDIDATE_K=12` keeps cross-encoder reranking but scores fewer candidates than the old 30-candidate path.
- `GOOGLE_CLOUD_EMBED_LOCATION=us-central1` keeps query embeddings regional even if `GOOGLE_CLOUD_LOCATION=global` is used for Gemini 3.1 models.
- `ENABLE_LATENCY_DEBUG=true` logs stage timings (`latency.retrieve`, `latency.tool`, `latency.answer`, `latency.agent`) and returns `latency_trace` in `/chat`.
- `/chat/stream` streams answer tokens to the browser, so the visible TTFT is measured when the first token arrives instead of after the full JSON response completes.
- `VERTEX_MODEL=gemini-2.5-flash` + `VERTEX_THINKING_BUDGET=0` disables thinking entirely on Flash, giving sub-second TTFT. (Gemini 2.5 Pro rejects `thinking_budget=0` — its valid range is 128–32,768. Gemini 3 Pro uses `thinking_level=LOW|MEDIUM|HIGH` instead, with no `thinking_budget`.)
- `ANSWER_CONTEXT_K=6`, `ANSWER_CONTEXT_CHARS=1400`, and `ANSWER_MAX_OUTPUT_TOKENS=1600` keep answer quality reasonable while still bounding prompt/output size.
