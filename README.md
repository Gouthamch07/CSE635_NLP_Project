# UB CSE Website Chatbot

Conversational assistant for the University at Buffalo Computer Science & Engineering
department, grounded in the department's website plus linked PDFs/syllabi.

## Sub-systems

| Module | Purpose |
| --- | --- |
| `ub_cse_bot/scraper` | Recursive crawl (crawl4ai + httpx fallback, Playwright for JS), PDF extraction |
| `ub_cse_bot/kg` | Entity & relation extraction (Course / Faculty / Program / Lab) + Neo4j loader |
| `ub_cse_bot/embeddings` | Vertex `text-embedding-004`, **lazy** disk cache, **contextual** (breadcrumb-prefixed) chunking |
| `ub_cse_bot/rag` | Pinecone dense + BM25 sparse + **RRF fusion** + lexical rerank + **BGE cross-encoder** rerank |
| `ub_cse_bot/agent` | Tool-using orchestrator over Vertex Gemini 2.5 Pro with KG tools + retrieve tool |
| `ub_cse_bot/guardrails` | Two-stage (keyword + LLM) scope classifier with polite redirects |
| `ub_cse_bot/dialogue` | Short-term rolling memory + opt-in personalized memory (facts + answer cache) |
| `ub_cse_bot/ui` | Streamlit chat UI with a live retrieval/rerank log panel |
| `ub_cse_bot/eval` | RAGAS (faithfulness, relevancy, precision), Hit@K / MRR, latency, robustness |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium            # for crawl4ai

cp .env.example .env                   # fill in Vertex / Pinecone / Neo4j creds
docker compose -f docker/docker-compose.yml up -d   # Neo4j
```

Set:
- `GOOGLE_APPLICATION_CREDENTIALS` to a Vertex service-account JSON
- `PINECONE_API_KEY`
- `NEO4J_PASSWORD`

## Pipeline

```bash
# 1. Crawl UB CSE
python scripts/run_crawl.py            # -> data/raw/corpus.jsonl

# 2. Build knowledge graph (Neo4j + JSON snapshot)
python scripts/build_kg.py             # -> data/processed/kg.json

# 3. Build hybrid index (BM25 on disk + Pinecone upsert w/ contextual embeddings)
python scripts/build_index.py          # -> data/processed/bm25.pkl + Pinecone

# 4. Launch UI
streamlit run ub_cse_bot/ui/app.py

# 5. Evaluation
python scripts/run_eval.py             # -> data/eval_out/*.json
```

## Notes on the bonus features

- **Course -> related labs/faculty**: `agent.tools.related_labs` walks
  `(Course)<-[:TAUGHT_BY]-(Faculty)-[:MEMBER_OF_LAB]->(Lab)` in Neo4j.
- **Dual rerank**: `LexicalReranker` (token overlap boost for codes like `CSE 574`)
  feeds a `CrossEncoderReranker` (BGE `bge-reranker-v2-m3`). Both stages are logged
  in the Streamlit backend panel with per-stage scores.
- **Personalized memory**: toggle "Remember me…" in the sidebar. The agent stores
  salient facts and caches common answers to cut TTFT on repeat queries.
- **Lazy embeddings**: `LazyEmbedder` keys on `sha1(model | text)` and caches to
  `data/emb_cache/`, so re-crawls only embed new/changed chunks.
- **Contextual embeddings**: each chunk is prefixed with `[Page] / [Section] /
  [Summary]` before embedding, while the clean chunk is what the LLM sees.

## Tests

```bash
pytest -q
```
