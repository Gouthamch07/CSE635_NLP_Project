"""Entrypoint: crawl UB CSE and write raw corpus to data/raw/corpus.jsonl."""
from __future__ import annotations

from pathlib import Path

from config import get_settings
from ub_cse_bot.scraper import UBCSECrawler


def main() -> None:
    s = get_settings()
    out = s.data_dir / "raw" / "corpus.jsonl"
    crawler = UBCSECrawler()
    recs = crawler.crawl_sync()
    crawler.dump(recs, out)
    print(f"[ok] {len(recs)} docs -> {out}")


if __name__ == "__main__":
    main()
