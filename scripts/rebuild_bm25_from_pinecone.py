"""Rebuild data/processed/bm25.pkl from Pinecone metadata (no re-crawl).

Self-contained: imports only pinecone-client + rank-bm25 + stdlib so it runs
without the heavier ub_cse_bot deps (orjson, vertex SDK, sentence-transformers).
The pickle format matches what ub_cse_bot.rag.bm25.BM25Index.load() expects.
"""
from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

from pinecone import Pinecone
from rank_bm25 import BM25Okapi

NAMESPACE = "default"
FETCH_BATCH = 100

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def main() -> None:
    api_key = os.environ["PINECONE_API_KEY"]
    index_name = os.environ.get("PINECONE_INDEX", "ub-cse-chatbot")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    page_ids: list[str] = []
    for id_batch in index.list(namespace=NAMESPACE):
        page_ids.extend(id_batch)
    print(f"Discovered {len(page_ids)} ids in namespace={NAMESPACE}")

    ids: list[str] = []
    docs: list[str] = []
    meta: list[dict] = []

    for i in range(0, len(page_ids), FETCH_BATCH):
        chunk_ids = page_ids[i : i + FETCH_BATCH]
        res = index.fetch(ids=chunk_ids, namespace=NAMESPACE)
        vectors = res.vectors if hasattr(res, "vectors") else res["vectors"]
        for vid, v in vectors.items():
            md = v.get("metadata", {}) if isinstance(v, dict) else (v.metadata or {})
            text = md.get("text", "")
            if not text:
                continue
            ids.append(vid)
            docs.append(text)
            meta.append(
                {
                    "doc_id": md.get("doc_id", ""),
                    "url": md.get("url", ""),
                    "title": md.get("title", ""),
                    "section": md.get("section", ""),
                    "content_type": md.get("content_type", ""),
                    "text": text,
                }
            )
        print(f"Fetched {len(ids)}/{len(page_ids)}")

    print(f"Fitting BM25 over {len(docs)} docs...")
    bm25 = BM25Okapi([_tokenize(d) for d in docs])

    out = Path("data/processed/bm25.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump({"ids": ids, "docs": docs, "meta": meta, "bm25": bm25}, f)
    print(f"Saved BM25 -> {out}")


if __name__ == "__main__":
    main()
