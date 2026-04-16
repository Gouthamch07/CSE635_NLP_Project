from __future__ import annotations

from pathlib import Path

import pdfplumber


def extract_pdf_text(path: Path | str) -> str:
    path = Path(path)
    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                pages.append(txt)
    return "\n\n".join(pages).strip()
