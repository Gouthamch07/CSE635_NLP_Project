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
    """Vertex AI Gemini wrapper using the current google-genai SDK.

    The newer SDK exposes the controls we need for latency, especially streaming
    and thinking budget/level. The old vertexai.generative_models path is
    deprecated and did not expose those controls in this environment.
    """

    def __init__(self, model: str | None = None) -> None:
        self.s = get_settings()
        self.model_name = model or self.s.vertex_model
        self._client = None

    def _load(self):
        if self._client is not None:
            return self._client
        from google import genai
        from google.genai.types import HttpOptions

        self._client = genai.Client(
            vertexai=True,
            project=self.s.google_cloud_project,
            location=self.s.google_cloud_location,
            http_options=HttpOptions(api_version="v1"),
        )
        log.info(
            "vertex.gemini loaded sdk=google-genai model=%s project=%s location=%s "
            "thinking_budget=%s thinking_level=%s",
            self.model_name,
            self.s.google_cloud_project,
            self.s.google_cloud_location,
            self.s.vertex_thinking_budget,
            self.s.vertex_thinking_level or "(unset)",
        )
        return self._client

    def warmup(self) -> None:
        """Load the Vertex client object before the first user request."""
        self._load()

    # ---------- conversion helpers ----------
    @staticmethod
    def _split_messages(messages: Sequence[LLMMessage]) -> tuple[str | None, list]:
        from google.genai import types

        system_parts = [m.content for m in messages if m.role == "system"]
        contents = []
        for m in messages:
            if m.role == "system":
                continue
            role = "model" if m.role == "model" else "user"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m.content)]))
        return "\n\n".join(system_parts) or None, contents

    def _config(
        self,
        temperature: float,
        max_output_tokens: int,
        system_instruction: str | None = None,
        tools: list | None = None,
    ):
        from google.genai import types

        thinking_config = None
        level = self.s.vertex_thinking_level.strip().upper()
        if level:
            thinking_level = getattr(types.ThinkingLevel, level, None)
            if thinking_level is not None:
                thinking_config = types.ThinkingConfig(thinking_level=thinking_level)
        elif self.s.vertex_thinking_budget > 0 or self.s.vertex_thinking_budget == -1:
            thinking_config = types.ThinkingConfig(thinking_budget=self.s.vertex_thinking_budget)

        return types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
            thinking_config=thinking_config,
            tools=tools,
        )

    # ---------- calls ----------
    def generate(
        self,
        messages: Sequence[LLMMessage],
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
        tools: list | None = None,
        ) -> str:
        client = self._load()
        system, contents = self._split_messages(messages)
        cfg = self._config(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system,
            tools=tools,
        )

        t0 = time.perf_counter()
        resp = client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=cfg,
        )
        log.info("gemini.generate dt=%.2fs", time.perf_counter() - t0)
        return getattr(resp, "text", "") or ""

    def stream(
        self,
        messages: Sequence[LLMMessage],
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> Iterator[str]:
        """Token stream for real TTFT behavior in the UI."""
        client = self._load()
        system, contents = self._split_messages(messages)
        cfg = self._config(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system,
        )

        for chunk in client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=cfg,
        ):
            text = getattr(chunk, "text", "") or ""
            if text:
                yield text
