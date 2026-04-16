from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from config import get_settings

from ..utils.io import sha1, write_jsonl
from ..utils.logging import get_logger
from .cleaner import clean_html_to_markdown
from .pdf import extract_pdf_text

log = get_logger(__name__)


@dataclass
class CrawlRecord:
    url: str
    title: str
    text: str
    content_type: str
    depth: int
    parent: str | None
    outlinks: list[str] = field(default_factory=list)
    doc_id: str = ""

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = sha1(self.url)


class UBCSECrawler:
    """Recursive crawler for UB CSE — uses crawl4ai when available, httpx fallback.

    crawl4ai gives us JS-rendered content via Playwright; the httpx path is a
    lightweight fallback so the module stays importable without browsers.
    """

    def __init__(
        self,
        seeds: Iterable[str] | None = None,
        max_depth: int | None = None,
        max_pages: int | None = None,
        concurrency: int | None = None,
    ) -> None:
        s = get_settings()
        self.seeds = list(seeds) if seeds else [s.crawl_root]
        self.max_depth = max_depth if max_depth is not None else s.crawl_max_depth
        self.max_pages = max_pages if max_pages is not None else s.crawl_max_pages
        self.concurrency = concurrency if concurrency is not None else s.crawl_concurrency
        self.allowed = set(s.allowed_domains)
        self.pdf_dir = s.data_dir / "pdf"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    # ---------- helpers ----------
    def _in_scope(self, url: str) -> bool:
        try:
            host = urlparse(url).netloc.lower()
        except Exception:
            return False
        return any(host == d or host.endswith("." + d) for d in self.allowed)

    def _extract_links(self, html: str, base: str) -> list[str]:
        out: list[str] = []
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"].split("#", 1)[0].strip()
            if not href or href.startswith(("javascript:", "mailto:", "tel:")):
                continue
            absu = urljoin(base, href)
            if absu.startswith(("http://", "https://")) and self._in_scope(absu):
                out.append(absu)
        # dedupe preserving order
        seen: set[str] = set()
        return [u for u in out if not (u in seen or seen.add(u))]

    def _title(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        return h1.get_text(strip=True) if h1 else ""

    # ---------- crawl4ai path ----------
    async def _crawl_with_crawl4ai(self) -> list[CrawlRecord]:
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
            from crawl4ai.async_configs import BrowserConfig
        except Exception as exc:  # pragma: no cover - optional dep
            log.warning("crawl4ai unavailable (%s); falling back to httpx", exc)
            return await self._crawl_with_httpx()

        browser = BrowserConfig(headless=True, java_script_enabled=True)
        run_cfg = CrawlerRunConfig(
            word_count_threshold=30,
            exclude_external_links=True,
            remove_overlay_elements=True,
            process_iframes=False,
        )

        visited: set[str] = set()
        queue: list[tuple[str, int, str | None]] = [(u, 0, None) for u in self.seeds]
        results: list[CrawlRecord] = []

        sem = asyncio.Semaphore(self.concurrency)
        async with AsyncWebCrawler(config=browser) as crawler:

            async def fetch(url: str, depth: int, parent: str | None) -> None:
                if url in visited or len(results) >= self.max_pages:
                    return
                visited.add(url)
                async with sem:
                    try:
                        res = await crawler.arun(url=url, config=run_cfg)
                    except Exception as exc:
                        log.warning("crawl4ai failed %s: %s", url, exc)
                        return
                if not res.success or not res.html:
                    return
                markdown = res.markdown or clean_html_to_markdown(res.html, url)
                rec = CrawlRecord(
                    url=url,
                    title=self._title(res.html),
                    text=markdown,
                    content_type="text/html",
                    depth=depth,
                    parent=parent,
                    outlinks=self._extract_links(res.html, url),
                )
                results.append(rec)
                if depth < self.max_depth:
                    for link in rec.outlinks:
                        if link not in visited:
                            queue.append((link, depth + 1, url))

            while queue and len(results) < self.max_pages:
                batch = queue[: self.concurrency]
                queue = queue[self.concurrency :]
                await asyncio.gather(*(fetch(u, d, p) for u, d, p in batch))

        # PDFs found among outlinks
        pdf_links = {l for r in results for l in r.outlinks if l.lower().endswith(".pdf")}
        results += await self._fetch_pdfs(pdf_links)
        return results

    # ---------- httpx fallback ----------
    async def _crawl_with_httpx(self) -> list[CrawlRecord]:
        visited: set[str] = set()
        queue: list[tuple[str, int, str | None]] = [(u, 0, None) for u in self.seeds]
        results: list[CrawlRecord] = []
        sem = asyncio.Semaphore(self.concurrency)

        headers = {"User-Agent": "UB-CSE-ChatbotCrawler/0.1 (+academic project)"}
        async with httpx.AsyncClient(
            headers=headers, follow_redirects=True, timeout=30.0
        ) as client:

            async def fetch(url: str, depth: int, parent: str | None) -> None:
                if url in visited or len(results) >= self.max_pages:
                    return
                visited.add(url)
                async with sem:
                    try:
                        r = await client.get(url)
                    except Exception as exc:
                        log.warning("httpx failed %s: %s", url, exc)
                        return
                if r.status_code != 200:
                    return
                ctype = r.headers.get("content-type", "").lower()
                if "pdf" in ctype or url.lower().endswith(".pdf"):
                    pdf_path = self.pdf_dir / f"{sha1(url)}.pdf"
                    pdf_path.write_bytes(r.content)
                    text = extract_pdf_text(pdf_path)
                    results.append(
                        CrawlRecord(
                            url=url,
                            title=Path(urlparse(url).path).name,
                            text=text,
                            content_type="application/pdf",
                            depth=depth,
                            parent=parent,
                        )
                    )
                    return
                if "html" not in ctype:
                    return
                html = r.text
                rec = CrawlRecord(
                    url=url,
                    title=self._title(html),
                    text=clean_html_to_markdown(html, url),
                    content_type="text/html",
                    depth=depth,
                    parent=parent,
                    outlinks=self._extract_links(html, url),
                )
                results.append(rec)
                if depth < self.max_depth:
                    for link in rec.outlinks:
                        if link not in visited:
                            queue.append((link, depth + 1, url))

            while queue and len(results) < self.max_pages:
                batch = queue[: self.concurrency]
                queue = queue[self.concurrency :]
                await asyncio.gather(*(fetch(u, d, p) for u, d, p in batch))
        return results

    async def _fetch_pdfs(self, urls: set[str]) -> list[CrawlRecord]:
        if not urls:
            return []
        out: list[CrawlRecord] = []
        async with httpx.AsyncClient(
            timeout=60.0, follow_redirects=True,
            headers={"User-Agent": "UB-CSE-ChatbotCrawler/0.1"},
        ) as client:
            for url in urls:
                try:
                    r = await client.get(url)
                    if r.status_code != 200:
                        continue
                    pdf_path = self.pdf_dir / f"{sha1(url)}.pdf"
                    pdf_path.write_bytes(r.content)
                    text = extract_pdf_text(pdf_path)
                    out.append(
                        CrawlRecord(
                            url=url,
                            title=Path(urlparse(url).path).name,
                            text=text,
                            content_type="application/pdf",
                            depth=0,
                            parent=None,
                        )
                    )
                except Exception as exc:
                    log.warning("pdf fetch failed %s: %s", url, exc)
        return out

    # ---------- public ----------
    async def crawl(self) -> list[CrawlRecord]:
        log.info("Crawling %s (max_depth=%d, max_pages=%d)",
                 self.seeds, self.max_depth, self.max_pages)
        recs = await self._crawl_with_crawl4ai()
        log.info("Crawl done: %d docs", len(recs))
        return recs

    def crawl_sync(self) -> list[CrawlRecord]:
        return asyncio.run(self.crawl())

    def dump(self, recs: list[CrawlRecord], path: Path) -> None:
        rows = [rec.__dict__ for rec in recs]
        write_jsonl(path, rows)
        log.info("Wrote %d records to %s", len(rows), path)
