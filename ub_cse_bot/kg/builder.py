from __future__ import annotations

from pathlib import Path

from ..utils.io import read_jsonl, write_json
from ..utils.logging import get_logger
from .extractor import EntityExtractor
from .neo4j_store import Neo4jStore

log = get_logger(__name__)


def build_knowledge_graph(corpus_path: Path, to_neo4j: bool = True) -> Path:
    """Extract entities from the crawl corpus and (optionally) load to Neo4j.

    Also writes a JSON dump of the graph to data/processed/kg.json so the
    system remains usable without a running Neo4j instance.
    """
    docs = read_jsonl(corpus_path)
    log.info("Loaded %d docs from %s", len(docs), corpus_path)

    ex = EntityExtractor()
    for d in docs:
        ex.ingest(d)
    result = ex.result()

    out_path = corpus_path.parent.parent / "processed" / "kg.json"
    payload = {
        "courses": [c.__dict__ for c in result.courses.values()],
        "faculty": [f.__dict__ for f in result.faculty.values()],
        "programs": [p.__dict__ for p in result.programs.values()],
        "labs": [l.__dict__ for l in result.labs.values()],
        "edges": [e.__dict__ for e in result.edges],
    }
    write_json(out_path, payload)
    log.info("Wrote KG snapshot to %s", out_path)

    if to_neo4j:
        try:
            store = Neo4jStore()
            store.ensure_constraints()
            store.upsert_courses(result.courses.values())
            store.upsert_faculty(result.faculty.values())
            store.upsert_programs(result.programs.values())
            store.upsert_labs(result.labs.values())
            store.upsert_edges(result.edges)
            store.close()
            log.info("Loaded KG into Neo4j")
        except Exception as exc:
            log.warning("Neo4j load skipped: %s", exc)

    return out_path
