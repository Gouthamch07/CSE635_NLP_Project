from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Sequence

from rank_bm25 import BM25Okapi

from ..utils.logging import get_logger

log = get_logger(__name__)


_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class BM25Index:
    """Sparse/lexical retrieval — precise for course codes, names, and acronyms."""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.docs: list[str] = []
        self.meta: list[dict] = []
        self._bm25: BM25Okapi | None = None

    def fit(self, ids: Sequence[str], docs: Sequence[str], meta: Sequence[dict]) -> None:
        self.ids = list(ids)
        self.docs = list(docs)
        self.meta = list(meta)
        self._bm25 = BM25Okapi([_tokenize(d) for d in self.docs])
        log.info("BM25 fitted over %d docs", len(self.docs))

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "id": self.ids[i],
                "score": float(scores[i]),
                "text": self.docs[i],
                "metadata": self.meta[i],
            }
            for i in idx
            if scores[i] > 0
        ]

    # ---------- persistence ----------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {"ids": self.ids, "docs": self.docs, "meta": self.meta, "bm25": self._bm25},
                f,
            )

    def load(self, path: Path) -> None:
        with path.open("rb") as f:
            blob = pickle.load(f)
        self.ids = blob["ids"]
        self.docs = blob["docs"]
        self.meta = blob["meta"]
        self._bm25 = blob["bm25"]
