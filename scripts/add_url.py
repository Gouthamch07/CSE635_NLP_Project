"""Add a single URL to the live Pinecone + BM25 index.

Usage:
    PYTHONPATH=. python3 scripts/add_url.py "https://cse.buffalo.edu/~kelinluo/teaching/cse531A-fall25/syllabus/index.html"

Fetches the page, extracts clean text, chunks it, embeds with Vertex
text-embedding-004, and upserts into Pinecone. Then rebuilds the local
BM25 pickle so sparse retrieval also sees the new content.
"""
from __future__ import annotations

import hashlib
import pickle
import re
import sys
import textwrap
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

import os

NAMESPACE = "default"
CHUNK_WORDS = 280
OVERLAP = 40
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def fetch(url: str) -> tuple[str, str]:
    r = httpx.get(url, timeout=20, follow_redirects=True,
                  headers={"User-Agent": "ub-cse-chatbot/1.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else url
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    body = soup.body or soup
    text = body.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


def chunk(text: str) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + CHUNK_WORDS]))
        i += CHUNK_WORDS - OVERLAP
    return [c for c in chunks if c.strip()]


def vertex_embed(texts: list[str]) -> list[list[float]]:
    import vertexai
    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    vertexai.init(project=project, location=location)
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    out: list[list[float]] = []
    BATCH = 16
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        inputs = [TextEmbeddingInput(t, "RETRIEVAL_DOCUMENT") for t in batch]
        embs = model.get_embeddings(inputs)
        out.extend([e.values for e in embs])
        print(f"embedded {len(out)}/{len(texts)}")
    return out


def main(url: str) -> None:
    print(f"fetching {url}")
    title, text = fetch(url)
    chunks = chunk(text)
    print(f"got {len(chunks)} chunks (title='{title[:80]}')")
    if not chunks:
        print("nothing to ingest"); return

    doc_id = hashlib.sha1(url.encode()).hexdigest()
    ids   = [f"{doc_id}__{i:04d}" for i in range(len(chunks))]
    metas = [
        {
            "doc_id": doc_id,
            "url": url,
            "title": title,
            "section": title,
            "content_type": "text/html",
            "text": c[:1500],
        }
        for c in chunks
    ]

    vectors = vertex_embed(chunks)

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ.get("PINECONE_INDEX", "ub-cse-chatbot"))
    payload = [{"id": i, "values": v, "metadata": m}
               for i, v, m in zip(ids, vectors, metas)]
    BATCH = 50
    for i in range(0, len(payload), BATCH):
        index.upsert(vectors=payload[i : i + BATCH], namespace=NAMESPACE)
    print(f"upserted {len(payload)} vectors to Pinecone")

    bm25_path = Path("data/processed/bm25.pkl")
    if bm25_path.exists():
        with bm25_path.open("rb") as f:
            blob = pickle.load(f)
        # extend
        blob["ids"]  = list(blob["ids"])  + ids
        blob["docs"] = list(blob["docs"]) + chunks
        blob["meta"] = list(blob["meta"]) + metas
        blob["bm25"] = BM25Okapi(
            [[t.lower() for t in _TOKEN.findall(d)] for d in blob["docs"]]
        )
        with bm25_path.open("wb") as f:
            pickle.dump(blob, f)
        print(f"BM25 refit over {len(blob['docs'])} docs -> {bm25_path}")

    print("done. Restart streamlit/uvicorn so the new BM25 is loaded.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(textwrap.dedent(__doc__).strip()); sys.exit(2)
    main(sys.argv[1])
