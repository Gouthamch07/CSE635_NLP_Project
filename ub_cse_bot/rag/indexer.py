from __future__ import annotations

from pathlib import Path

from config import get_settings

from ..embeddings.contextual import ContextualChunker
from ..embeddings.lazy import LazyEmbedder
from ..utils.io import read_jsonl
from ..utils.logging import get_logger
from .bm25 import BM25Index
from .pinecone_store import PineconeStore

log = get_logger(__name__)


def build_index(
    corpus_path: Path,
    bm25_out: Path,
    namespace: str = "default",
    push_pinecone: bool = True,
) -> None:
    """Chunk -> contextual-embed -> Pinecone upsert + BM25 fit on-disk."""
    docs = read_jsonl(corpus_path)
    chunker = ContextualChunker()
    chunks = []
    for d in docs:
        chunks.extend(chunker.chunk_doc(d))
    log.info("Chunked %d docs into %d chunks", len(docs), len(chunks))

    ids = [c.chunk_id for c in chunks]
    metas = [
        {
            "doc_id": c.doc_id,
            "url": c.url,
            "title": c.title,
            "section": c.section,
            "content_type": c.meta.get("content_type", ""),
            "text": c.text[:1500],   # Pinecone metadata size limit
        }
        for c in chunks
    ]

    # ---- BM25 on clean text ----
    bm25 = BM25Index()
    bm25.fit(ids, [c.text for c in chunks], metas)
    bm25.save(bm25_out)
    log.info("BM25 persisted -> %s", bm25_out)

    # ---- Contextual embeddings + Pinecone ----
    if not push_pinecone:
        return

    s = get_settings()
    embedder = LazyEmbedder(cache_dir=s.data_dir / "emb_cache")
    vectors = embedder.embed_documents([c.contextualized_text for c in chunks])

    store = PineconeStore()
    store.ensure_index()
    store.upsert(ids, vectors, metas, namespace=namespace)
    log.info("Indexed %d chunks to Pinecone namespace=%s", len(chunks), namespace)
