from __future__ import annotations

import re

import trafilatura
from bs4 import BeautifulSoup


_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{3,}")


def clean_html_to_markdown(html: str, url: str = "") -> str:
    """Return readable markdown-ish text from HTML, stripping chrome."""
    extracted = trafilatura.extract(
        html,
        url=url or None,
        include_comments=False,
        include_tables=True,
        include_formatting=True,
        favor_recall=True,
        output_format="markdown",
    )
    if extracted:
        return _normalize(extracted)

    # Fallback: soup text with header hints
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "footer", "nav", "aside"]):
        tag.decompose()
    parts: list[str] = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name.startswith("h"):
            level = int(el.name[1])
            parts.append(f"{'#' * level} {text}")
        else:
            parts.append(text)
    return _normalize("\n\n".join(parts))


def _normalize(text: str) -> str:
    text = _WS.sub(" ", text)
    text = _NL.sub("\n\n", text)
    return text.strip()
