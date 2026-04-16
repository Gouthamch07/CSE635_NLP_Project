"""Build BM25 + Pinecone indices from the crawl corpus."""
from __future__ import annotations

from config import get_settings
from ub_cse_bot.rag import build_index


def main() -> None:
    s = get_settings()
    corpus = s.data_dir / "raw" / "corpus.jsonl"
    bm25_out = s.data_dir / "processed" / "bm25.pkl"
    build_index(corpus, bm25_out=bm25_out, namespace="default", push_pinecone=True)


if __name__ == "__main__":
    main()
