from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ..llm.vertex_client import LLMMessage, VertexGemini
from ..utils.logging import get_logger

log = get_logger(__name__)

Decision = Literal["in_scope", "small_talk", "out_of_scope"]

_IN_KEYWORDS = (
    "cse", "ub", "buffalo", "course", "class", "prerequisite", "faculty",
    "professor", "ta ", "syllabus", "credit", "ms", "phd", "bs ",
    "degree", "program", "research", "lab", "office hours",
    "catalog", "registrar", "graduate", "undergraduate",
)
_SMALLTALK = (
    "hello", "hi", "hey", "thanks", "thank you", "good morning",
    "good afternoon", "bye", "goodbye", "how are you", "what's up",
)
_OUT_BLOCKLIST = (
    "pizza", "weather", "stock", "crypto", "movie", "lyrics", "recipe",
)


@dataclass
class ScopeDecision:
    label: Decision
    reason: str
    redirect: str = ""


class ScopeClassifier:
    """Two-stage guardrail:
       1) Cheap regex/keyword layer (covers ~80% of obvious cases).
       2) LLM fallback with a constrained prompt for the ambiguous middle.
    """

    def __init__(self, llm: VertexGemini | None = None) -> None:
        self.llm = llm

    def _keyword_decision(self, q: str) -> ScopeDecision | None:
        t = q.lower().strip()
        if any(k in t for k in _OUT_BLOCKLIST):
            return ScopeDecision(
                label="out_of_scope",
                reason="matched out-of-scope keyword",
                redirect=_redirect(),
            )
        if any(re.search(rf"\b{re.escape(k)}\b", t) for k in _SMALLTALK) and len(t) < 40:
            return ScopeDecision(label="small_talk", reason="short greeting / pleasantry")
        if any(k in t for k in _IN_KEYWORDS):
            return ScopeDecision(label="in_scope", reason="matched CSE keyword")
        return None

    def classify(self, query: str) -> ScopeDecision:
        kw = self._keyword_decision(query)
        if kw is not None:
            return kw

        if self.llm is None:
            # Conservative default: treat unknown as in_scope but flag
            return ScopeDecision(
                label="in_scope",
                reason="no keyword hit; no LLM available; default to answer",
            )

        system = (
            "You are a strict classifier for a University at Buffalo CSE department "
            "chatbot. Classify the user's message as one of: "
            "in_scope (CSE programs, courses, faculty, research, policies), "
            "small_talk (greetings/pleasantries), or out_of_scope (everything else). "
            "Respond with only the label."
        )
        msg = [
            LLMMessage("system", system),
            LLMMessage("user", query),
        ]
        try:
            label = self.llm.generate(msg, temperature=0.0, max_output_tokens=8).strip().lower()
        except Exception as exc:
            log.warning("scope LLM failed: %s", exc)
            return ScopeDecision(label="in_scope", reason="llm failed; default allow")
        if label not in {"in_scope", "small_talk", "out_of_scope"}:
            label = "in_scope"
        return ScopeDecision(
            label=label,  # type: ignore[arg-type]
            reason="llm-classified",
            redirect=_redirect() if label == "out_of_scope" else "",
        )


def _redirect() -> str:
    return (
        "I'm focused on the University at Buffalo CSE department — "
        "programs, courses, faculty, and research. "
        "Ask me about a CSE course, degree requirement, or faculty member, "
        "and I'll do my best to help."
    )
