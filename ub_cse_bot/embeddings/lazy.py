from __future__ import annotations

from pathlib import Path
from typing import Sequence

import orjson

from ..utils.io import sha1
from ..utils.logging import get_logger
from .vertex import VertexEmbedder

log = get_logger(__name__)


class LazyEmbedder:
    """Disk-backed cache that only calls Vertex for texts we haven't seen.

    Why "lazy":
      - Upstream embedding calls are the costliest part of indexing.
      - Re-crawls mostly return the same chunks; we should not re-embed them.
      - We key on sha1(text + '|' + model) so model changes invalidate the cache.
    """

    def __init__(self, cache_dir: Path, embedder: VertexEmbedder | None = None) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or VertexEmbedder()
        self._mem: dict[str, list[float]] = {}

    def _key(self, text: str) -> str:
        return sha1(f"{self.embedder.model_name}|{text}")

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key[:2]}" / f"{key}.json"

    def _get_cached(self, key: str) -> list[float] | None:
        if key in self._mem:
            return self._mem[key]
        p = self._path(key)
        if p.exists():
            vec = orjson.loads(p.read_bytes())
            self._mem[key] = vec
            return vec
        return None

    def _put(self, key: str, vec: list[float]) -> None:
        self._mem[key] = vec
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(orjson.dumps(vec))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        vectors: list[list[float] | None] = [self._get_cached(k) for k in keys]

        misses = [(i, texts[i]) for i, v in enumerate(vectors) if v is None]
        if misses:
            log.info("Lazy embed: %d/%d cache miss", len(misses), len(texts))
            fresh = self.embedder.embed([t for _, t in misses], task_type="RETRIEVAL_DOCUMENT")
            for (idx, _), vec in zip(misses, fresh):
                self._put(keys[idx], vec)
                vectors[idx] = vec
        else:
            log.info("Lazy embed: 0 cache miss (%d cached)", len(texts))
        return vectors  # type: ignore[return-value]

    def embed_query(self, text: str) -> list[float]:
        # Queries are rarely cache-hits, but key them anyway.
        key = self._key("Q|" + text)
        cached = self._get_cached(key)
        if cached is not None:
            return cached
        vec = self.embedder.embed_query(text)
        self._put(key, vec)
        return vec
