from .schema import Course, Faculty, Program, Lab, Edge
from .extractor import EntityExtractor

__all__ = [
    "Course", "Faculty", "Program", "Lab", "Edge",
    "EntityExtractor", "Neo4jStore", "build_knowledge_graph",
]


def __getattr__(name: str):
    # Lazy imports so `from ub_cse_bot.kg import EntityExtractor` works
    # without the `neo4j` package being installed.
    if name == "Neo4jStore":
        from .neo4j_store import Neo4jStore
        return Neo4jStore
    if name == "build_knowledge_graph":
        from .builder import build_knowledge_graph
        return build_knowledge_graph
    raise AttributeError(name)
