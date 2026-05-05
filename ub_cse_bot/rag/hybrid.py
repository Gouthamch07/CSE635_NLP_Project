from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import get_settings

from ..embeddings.lazy import LazyEmbedder
from ..utils.logging import get_logger
from .bm25 import BM25Index
from .pinecone_store import PineconeStore
from .reranker import (
    CrossEncoderReranker,
    LexicalReranker,
    RerankStep,
    reciprocal_rank_fusion,
)

log = get_logger(__name__)


@dataclass
class RetrievalHit:
    id: str
    text: str
    url: str
    title: str
    section: str
    score: float
    source: str = "hybrid"
    meta: dict = field(default_factory=dict)


@dataclass
class RetrievalTrace:
    query: str
    stages: list[RerankStep] = field(default_factory=list)
    hits: list[RetrievalHit] = field(default_factory=list)

    def add_stage(self, stage: str, cands: list[dict]) -> None:
        self.stages.append(
            RerankStep(
                stage=stage,
                scores=[
                    (c["id"], float(c.get("_ce_score") or c.get("_lex_score") or c.get("score") or 0))
                    for c in cands[:25]
                ],
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "stages": [
                {"stage": s.stage, "scores": s.scores} for s in self.stages
            ],
            "hits": [h.__dict__ for h in self.hits],
        }


class HybridRetriever:
    """Dense (Pinecone) + sparse (BM25) retrieval with RRF fusion, lexical tiebreak,
    and cross-encoder reranking.

    Usage:
        r = HybridRetriever.from_disk(bm25_path)
        trace = r.retrieve("Who teaches CSE 574?")
    """

    def __init__(
        self,
        bm25: BM25Index,
        pinecone: PineconeStore | None = None,
        embedder: LazyEmbedder | None = None,
        cross_encoder: CrossEncoderReranker | None = None,
        lexical: LexicalReranker | None = None,
        namespace: str = "default",
    ) -> None:
        self.s = get_settings()
        self.bm25 = bm25
        self.pinecone = pinecone or PineconeStore()
        self.embedder = embedder or LazyEmbedder(
            cache_dir=self.s.data_dir / "emb_cache"
        )
        self.cross_encoder = cross_encoder or CrossEncoderReranker()
        self.lexical = lexical or LexicalReranker()
        self.namespace = namespace

    # ---------- main entry ----------
    def retrieve(
        self,
        query: str,
        dense_k: int = 25,
        sparse_k: int = 25,
        final_k: int | None = None,
        filter: dict | None = None,
    ) -> RetrievalTrace:
        retrieve_t0 = time.perf_counter()
        final_k = final_k or self.s.rerank_top_k
        trace = RetrievalTrace(query=query)

        # 1. Dense
        dense_embed_ms = 0.0
        dense_query_ms = 0.0
        try:
            dense_embed_t0 = time.perf_counter()
            q_vec = self.embedder.embed_query(query)
            dense_embed_ms = (time.perf_counter() - dense_embed_t0) * 1000
            dense_query_t0 = time.perf_counter()
            dense = self.pinecone.query(q_vec, top_k=dense_k, namespace=self.namespace, filter=filter)
            dense_query_ms = (time.perf_counter() - dense_query_t0) * 1000
            for d in dense:
                md = d.get("metadata", {})
                d["text"] = md.get("text", "")
            trace.add_stage("dense", dense)
        except Exception as exc:
            log.warning("Dense retrieval failed: %s", exc)
            dense = []

        # 2. Sparse
        sparse_t0 = time.perf_counter()
        sparse = self.bm25.search(query, top_k=sparse_k)
        sparse_ms = (time.perf_counter() - sparse_t0) * 1000
        trace.add_stage("sparse", sparse)

        # 3. Fuse via RRF
        rrf_t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(dense, sparse)
        rrf_ms = (time.perf_counter() - rrf_t0) * 1000
        trace.add_stage("rrf_fusion", fused)

        # 4. Lexical rerank (cheap tie-break)
        lexical_t0 = time.perf_counter()
        lex = self.lexical.rerank(query, fused[: max(dense_k, sparse_k)])
        lexical_ms = (time.perf_counter() - lexical_t0) * 1000
        trace.add_stage("lexical_rerank", lex)

        # 5. Cross-encoder rerank (expensive, high-precision)
        ce_t0 = time.perf_counter()
        top_for_ce = lex[: min(self.s.cross_encoder_candidate_k, len(lex))]
        ce = self.cross_encoder.rerank(query, top_for_ce, top_k=final_k)
        ce_ms = (time.perf_counter() - ce_t0) * 1000
        trace.add_stage("cross_encoder_rerank", ce)

        # 6. Build hits
        trace.hits = [
            RetrievalHit(
                id=c["id"],
                text=c.get("text") or c.get("metadata", {}).get("text", ""),
                url=c.get("metadata", {}).get("url", ""),
                title=c.get("metadata", {}).get("title", ""),
                section=c.get("metadata", {}).get("section", ""),
                score=float(c.get("_ce_score") or c.get("score") or 0),
                source="hybrid",
                meta=c.get("metadata", {}),
            )
            for c in ce
        ]
        if self.s.enable_latency_debug:
            log.info(
                "latency.retrieve total_ms=%.1f dense_embed_ms=%.1f dense_query_ms=%.1f "
                "sparse_ms=%.1f rrf_ms=%.1f lexical_ms=%.1f cross_encoder_ms=%.1f "
                "dense_hits=%d sparse_hits=%d final_hits=%d",
                (time.perf_counter() - retrieve_t0) * 1000,
                dense_embed_ms,
                dense_query_ms,
                sparse_ms,
                rrf_ms,
                lexical_ms,
                ce_ms,
                len(dense),
                len(sparse),
                len(trace.hits),
            )
        return trace

    # ---------- loading ----------
    @classmethod
    def from_disk(cls, bm25_path: Path) -> "HybridRetriever":
        bm25 = BM25Index()
        bm25.load(bm25_path)
        return cls(bm25=bm25)

    def warmup(self, query: str = "MS in CSE credit requirements") -> None:
        """Prime embedding, Pinecone, and reranker clients before demo traffic.

        The Vertex embedder is warmed explicitly because LazyEmbedder may serve
        the warmup query from disk cache, leaving the underlying Vertex client
        uninitialized — and the first uncached user query then pays a ~10s
        cold-start cost.
        """
        if hasattr(self.embedder, "warmup"):
            self.embedder.warmup()
        self.retrieve(query, dense_k=5, sparse_k=5, final_k=min(3, self.s.rerank_top_k))
