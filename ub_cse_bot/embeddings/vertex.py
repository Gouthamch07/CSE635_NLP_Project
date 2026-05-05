from __future__ import annotations

from typing import Sequence

from config import get_settings

from ..utils.logging import get_logger

log = get_logger(__name__)


class VertexEmbedder:
    """Vertex AI text-embedding-004 wrapper (768-d)."""

    def __init__(self, model: str | None = None) -> None:
        self.s = get_settings()
        self.model_name = model or self.s.vertex_embed_model
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import vertexai
        from vertexai.language_models import TextEmbeddingModel

        location = self.s.google_cloud_embed_location or self.s.google_cloud_location
        vertexai.init(project=self.s.google_cloud_project, location=location)
        self._model = TextEmbeddingModel.from_pretrained(self.model_name)
        log.info(
            "vertex.embedder loaded model=%s project=%s location=%s",
            self.model_name,
            self.s.google_cloud_project,
            location,
        )

    def embed(
        self,
        texts: Sequence[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        self._load()
        from vertexai.language_models import TextEmbeddingInput

        inputs = [TextEmbeddingInput(text=t, task_type=task_type) for t in texts]
        out: list[list[float]] = []
        # Vertex caps batch size ~250; be conservative
        BATCH = 64
        for i in range(0, len(inputs), BATCH):
            resp = self._model.get_embeddings(inputs[i : i + BATCH])
            out.extend(e.values for e in resp)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], task_type="RETRIEVAL_QUERY")[0]

    def warmup(self) -> None:
        """Force Vertex SDK init + a real round-trip so the first user query
        doesn't pay auth / gRPC / model-resolution cost (~10s on cold start).
        """
        self._load()
        from vertexai.language_models import TextEmbeddingInput

        self._model.get_embeddings([TextEmbeddingInput(text="warmup", task_type="RETRIEVAL_QUERY")])
