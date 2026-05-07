# UB CSE Website Chatbot


https://www.loom.com/share/566e0f522a2d4818ab77754cffb6db0b

https://ub-cse-chatbot-682578067933.us-central1.run.app/

Conversational assistant for the University at Buffalo Computer Science & Engineering department, grounded in the department website plus linked PDFs/syllabi.

## Sub-systems

| Module | Purpose |
| --- | --- |
| `ub_cse_bot/scraper` | Recursive crawler using crawl4ai + Playwright for JavaScript-rendered pages, with an httpx fallback and PDF extraction |
| `ub_cse_bot/kg` | Heuristic Course / Faculty / Program / Lab extraction and Neo4j loading; the current extractor primarily builds prerequisite and teaching relations |
| `ub_cse_bot/embeddings` | Vertex `text-embedding-004`, lazy disk cache, and contextual chunking with page/section breadcrumbs |
| `ub_cse_bot/rag` | Pinecone dense retrieval + BM25 sparse retrieval + RRF fusion + lexical rerank + BGE cross-encoder rerank |
| `ub_cse_bot/agent` | Tool-using orchestrator over the configured Vertex Gemini model (`VERTEX_MODEL`) with retrieval and Neo4j-backed KG tools |
| `ub_cse_bot/guardrails` | Two-stage scope classifier with keyword and LLM checks for CSE-domain guardrails |
| `ub_cse_bot/dialogue` | Short-term rolling dialogue memory and opt-in personalized memory / answer cache |
| `ub_cse_bot/ui` | FastAPI-backed custom web UI with chat, personalization controls, theme toggle, and retrieval/rerank logs; Streamlit is available as a fallback/dev UI |
| `ub_cse_bot/eval` | Hit@K / MRR, guardrail accuracy, latency, RAGAS, and LLM-as-judge evaluation |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

cp .env.example .env

