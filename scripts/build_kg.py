"""Build the UB CSE knowledge graph from the crawl corpus."""
from __future__ import annotations

from config import get_settings
from ub_cse_bot.kg import build_knowledge_graph


def main() -> None:
    s = get_settings()
    corpus = s.data_dir / "raw" / "corpus.jsonl"
    build_knowledge_graph(corpus, to_neo4j=True)


if __name__ == "__main__":
    main()
