from __future__ import annotations

from typing import Iterable, Sequence

from config import get_settings

from ..utils.logging import get_logger

log = get_logger(__name__)


class PineconeStore:
    """Thin wrapper around Pinecone serverless index.

    Metadata stored per vector:
        doc_id, url, title, section, content_type, text (truncated)
    """

    def __init__(self, index: str | None = None) -> None:
        self.s = get_settings()
        self.name = index or self.s.pinecone_index
        self._pc = None
        self._index = None

    def _client(self):
        if self._pc is not None:
            return self._pc
        from pinecone import Pinecone

        self._pc = Pinecone(api_key=self.s.pinecone_api_key)
        return self._pc

    def ensure_index(self) -> None:
        from pinecone import ServerlessSpec

        pc = self._client()
        existing = {i["name"] for i in pc.list_indexes()}
        if self.name not in existing:
            log.info("Creating Pinecone index %s (dim=%d)", self.name, self.s.pinecone_dimension)
            pc.create_index(
                name=self.name,
                dimension=self.s.pinecone_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.s.pinecone_cloud, region=self.s.pinecone_region),
            )
        self._index = pc.Index(self.name)

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metadatas: Sequence[dict],
        namespace: str = "default",
    ) -> None:
        if self._index is None:
            self.ensure_index()
        vectors_payload = [
            {"id": i, "values": list(v), "metadata": m}
            for i, v, m in zip(ids, vectors, metadatas)
        ]
        BATCH = 100
        for i in range(0, len(vectors_payload), BATCH):
            self._index.upsert(vectors=vectors_payload[i : i + BATCH], namespace=namespace)
        log.info("Upserted %d vectors into %s/%s", len(vectors_payload), self.name, namespace)

    def query(
        self,
        vector: Sequence[float],
        top_k: int = 20,
        namespace: str = "default",
        filter: dict | None = None,
    ) -> list[dict]:
        if self._index is None:
            self.ensure_index()
        res = self._index.query(
            vector=list(vector),
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
            filter=filter,
        )
        return [
            {"id": m["id"], "score": m["score"], "metadata": m.get("metadata", {})}
            for m in res.get("matches", [])
        ]

    def delete_namespace(self, namespace: str = "default") -> None:
        if self._index is None:
            self.ensure_index()
        self._index.delete(delete_all=True, namespace=namespace)
