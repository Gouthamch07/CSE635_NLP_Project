from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Sequence

from config import get_settings

from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class LLMMessage:
    role: str           # "user" | "model" | "system"
    content: str


class VertexGemini:
    """Vertex AI Gemini wrapper with streaming + optional tool use.

    We route through google-cloud-aiplatform's GenerativeModel which supports
    Gemini 2.5 Pro / Flash on Vertex.
    """

    def __init__(self, model: str | None = None) -> None:
        self.s = get_settings()
        self.model_name = model or self.s.vertex_model
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=self.s.google_cloud_project, location=self.s.google_cloud_location)
        self._model = GenerativeModel(self.model_name)

    # ---------- conversion helpers ----------
    @staticmethod
    def _to_contents(messages: Sequence[LLMMessage]):
        from vertexai.generative_models import Content, Part
        # Vertex supports system via the model constructor; we inline it as a
        # user pre-message to stay portable across SDK versions.
        out = []
        for m in messages:
            role = "user" if m.role in ("system", "user") else "model"
            out.append(Content(role=role, parts=[Part.from_text(m.content)]))
        return out

    # ---------- calls ----------
    def generate(
        self,
        messages: Sequence[LLMMessage],
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        tools: list | None = None,
    ) -> str:
        self._load()
        from vertexai.generative_models import GenerationConfig

        cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_output_tokens)
        t0 = time.time()
        resp = self._model.generate_content(
            self._to_contents(messages),
            generation_config=cfg,
            tools=tools,
        )
        log.info("gemini.generate dt=%.2fs", time.time() - t0)
        try:
            return resp.text
        except Exception:
            return "".join(p.text for p in resp.candidates[0].content.parts if getattr(p, "text", None))

    def stream(
        self,
        messages: Sequence[LLMMessage],
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> Iterator[str]:
        """Token-stream — useful for keeping TTFT under 2s."""
        self._load()
        from vertexai.generative_models import GenerationConfig

        cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_output_tokens)
        stream = self._model.generate_content(
            self._to_contents(messages),
            generation_config=cfg,
            stream=True,
        )
        for chunk in stream:
            try:
                text = chunk.text
            except Exception:
                text = ""
            if text:
                yield text
