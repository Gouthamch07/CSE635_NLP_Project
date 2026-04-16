from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from ..utils.io import sha1
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ContextualChunk:
    chunk_id: str
    doc_id: str
    url: str
    title: str
    text: str                    # the original chunk text
    contextualized_text: str     # text prefixed with contextual summary for embedding
    breadcrumb: list[str] = field(default_factory=list)
    section: str = ""
    meta: dict = field(default_factory=dict)


class ContextualChunker:
    """Heading-aware chunker that prepends breadcrumb + section-summary to each
    chunk before embedding (Anthropic-style "contextual retrieval").

    The prepended context is used ONLY at embed/index time; the text stored
    alongside the vector (for the LLM) remains the clean chunk.
    """

    HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)

    def __init__(self, chunk_tokens: int = 280, overlap: int = 40) -> None:
        # approximate by words rather than real tokens; close enough for embed models
        self.chunk_words = chunk_tokens
        self.overlap = overlap

    # ---------- segmentation ----------
    def _segments(self, text: str) -> list[tuple[list[str], str]]:
        """Return list of (breadcrumb, body) by walking markdown-ish headings."""
        out: list[tuple[list[str], str]] = []
        stack: list[tuple[int, str]] = []
        last_end = 0
        for m in self.HEADING_RE.finditer(text):
            pre = text[last_end: m.start()].strip()
            if pre and stack:
                out.append(([h for _, h in stack], pre))
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            last_end = m.end()
        tail = text[last_end:].strip()
        if tail:
            out.append(([h for _, h in stack], tail))
        if not out:
            out.append(([], text.strip()))
        return out

    def _window(self, words: list[str]) -> Iterable[str]:
        step = self.chunk_words - self.overlap
        if step <= 0:
            step = self.chunk_words
        for i in range(0, len(words), step):
            piece = words[i : i + self.chunk_words]
            if not piece:
                continue
            yield " ".join(piece)

    # ---------- public ----------
    def chunk_doc(self, doc: dict) -> list[ContextualChunk]:
        url = doc.get("url", "")
        title = doc.get("title", "")
        doc_id = doc.get("doc_id") or sha1(url or title)
        chunks: list[ContextualChunk] = []

        for breadcrumb, body in self._segments(doc.get("text", "")):
            for piece in self._window(body.split()):
                section = " > ".join(breadcrumb) if breadcrumb else title
                context_prefix = self._context_prefix(title, breadcrumb, piece)
                contextualized = f"{context_prefix}\n\n{piece}"
                chunks.append(
                    ContextualChunk(
                        chunk_id=sha1(f"{doc_id}|{section}|{piece[:80]}"),
                        doc_id=doc_id,
                        url=url,
                        title=title,
                        text=piece,
                        contextualized_text=contextualized,
                        breadcrumb=breadcrumb,
                        section=section,
                        meta={"content_type": doc.get("content_type", "text/html")},
                    )
                )
        log.debug("Chunked %s into %d chunks", url, len(chunks))
        return chunks

    @staticmethod
    def _context_prefix(title: str, breadcrumb: list[str], piece: str) -> str:
        path = " > ".join(breadcrumb) if breadcrumb else title
        # short synthetic context; the embedding model uses this to anchor semantics
        summary = piece[:160].replace("\n", " ")
        return (
            f"[Page] {title}\n"
            f"[Section] {path}\n"
            f"[Summary] {summary}..."
        )
