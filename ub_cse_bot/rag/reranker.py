from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from config import get_settings

from ..utils.logging import get_logger

log = get_logger(__name__)

_TOKEN = re.compile(r"[A-Za-z0-9]+")


@dataclass
class RerankStep:
    stage: str
    scores: list[tuple[str, float]]  # (id, score)


class LexicalReranker:
    """Cheap lexical tie-breaker — boosts exact token overlap.

    Useful when a course-code query like "CSE 574" pulls a semantically
    similar but off-topic chunk ahead of the exact match.
    """

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        q_tokens = set(t.lower() for t in _TOKEN.findall(query))
        if not q_tokens:
            return candidates

        def boost(cand: dict) -> float:
            text = (cand.get("text") or cand.get("metadata", {}).get("text") or "").lower()
            tokens = set(_TOKEN.findall(text))
            overlap = len(q_tokens & tokens) / max(1, len(q_tokens))
            # preserve original score but add small overlap bonus
            return (cand.get("score") or 0.0) + 0.25 * overlap

        for c in candidates:
            c["_lex_score"] = boost(c)
        return sorted(candidates, key=lambda c: c["_lex_score"], reverse=True)


class CrossEncoderReranker:
    """BGE cross-encoder reranker. Loads the model lazily."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_settings().rerank_model
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(self.model_name, use_fp16=True)
        except Exception as exc:
            log.warning("FlagEmbedding unavailable (%s); using sentence-transformers fallback", exc)
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(
        self, query: str, candidates: list[dict], top_k: int = 8
    ) -> list[dict]:
        if not candidates:
            return []
        self._load()
        pairs = [
            [query, c.get("text") or c.get("metadata", {}).get("text", "")]
            for c in candidates
        ]
        # Both APIs expose compute_score / predict respectively.
        if hasattr(self._model, "compute_score"):
            raw = self._model.compute_score(pairs, normalize=True)
        else:
            raw = self._model.predict(pairs).tolist()
        if isinstance(raw, float):
            raw = [raw]
        for c, s in zip(candidates, raw):
            c["_ce_score"] = float(s)
        ranked = sorted(candidates, key=lambda c: c["_ce_score"], reverse=True)
        return ranked[:top_k]


def reciprocal_rank_fusion(
    *ranked_lists: Sequence[dict], k: int = 60
) -> list[dict]:
    """Standard RRF: score(id) = sum 1/(k + rank_i(id)) across lists."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    for lst in ranked_lists:
        for rank, cand in enumerate(lst):
            cid = cand["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            # keep richer payload if available
            if cid not in payloads or len(str(cand)) > len(str(payloads[cid])):
                payloads[cid] = cand
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for cid, sc in fused:
        cand = dict(payloads[cid])
        cand["score"] = sc
        out.append(cand)
    return out
