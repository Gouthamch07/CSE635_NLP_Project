from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..kg.neo4j_store import Neo4jStore
from ..rag.hybrid import HybridRetriever, RetrievalTrace
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class ToolResult:
    name: str
    payload: Any
    ok: bool = True
    error: str = ""
    trace: dict | None = None


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[dict], ToolResult]
    schema: dict = field(default_factory=dict)


def build_tool_registry(
    retriever: HybridRetriever,
    kg_store: Neo4jStore | None = None,
) -> dict[str, Tool]:
    """Tools the agent may call. Each returns a ToolResult with structured payload."""

    def _retrieve(args: dict) -> ToolResult:
        q = args.get("query", "").strip()
        if not q:
            return ToolResult("retrieve", None, ok=False, error="missing query")
        trace: RetrievalTrace = retriever.retrieve(q, final_k=args.get("k", 8))
        payload = {
            "hits": [
                {
                    "id": h.id, "url": h.url, "title": h.title,
                    "section": h.section, "score": h.score, "text": h.text,
                }
                for h in trace.hits
            ],
        }
        return ToolResult("retrieve", payload, trace=trace.to_dict())

    def _course_prereqs(args: dict) -> ToolResult:
        code = _norm_code(args.get("code", ""))
        if not kg_store or not code:
            return ToolResult("course_prereqs", [], ok=False, error="no code / no kg")
        try:
            prs = kg_store.prerequisites(code)
            return ToolResult("course_prereqs", {"code": code, "prerequisites": prs})
        except Exception as e:
            return ToolResult("course_prereqs", None, ok=False, error=str(e))

    def _course_faculty(args: dict) -> ToolResult:
        code = _norm_code(args.get("code", ""))
        if not kg_store or not code:
            return ToolResult("course_faculty", [], ok=False, error="no code / no kg")
        try:
            return ToolResult("course_faculty", {
                "code": code,
                "faculty": kg_store.related_faculty_for_course(code),
            })
        except Exception as e:
            return ToolResult("course_faculty", None, ok=False, error=str(e))

    def _related_labs(args: dict) -> ToolResult:
        """Bonus: course -> related research labs & faculty."""
        code = _norm_code(args.get("code", ""))
        if not kg_store or not code:
            return ToolResult("related_labs", [], ok=False, error="no code / no kg")
        try:
            return ToolResult("related_labs", {
                "code": code,
                "labs": kg_store.related_labs_for_course(code),
            })
        except Exception as e:
            return ToolResult("related_labs", None, ok=False, error=str(e))

    def _faculty_info(args: dict) -> ToolResult:
        name = (args.get("name") or "").strip()
        if not kg_store or not name:
            return ToolResult("faculty_info", {}, ok=False, error="no name / no kg")
        try:
            return ToolResult("faculty_info", kg_store.faculty_info(name))
        except Exception as e:
            return ToolResult("faculty_info", None, ok=False, error=str(e))

    def _lab_info(args: dict) -> ToolResult:
        name = (args.get("name") or "").strip()
        if not kg_store or not name:
            return ToolResult("lab_info", {}, ok=False, error="no name / no kg")
        try:
            return ToolResult("lab_info", kg_store.lab_info(name))
        except Exception as e:
            return ToolResult("lab_info", None, ok=False, error=str(e))

    def _program_info(args: dict) -> ToolResult:
        name = (args.get("name") or "").strip()
        if not kg_store or not name:
            return ToolResult("program_info", {}, ok=False, error="no name / no kg")
        try:
            return ToolResult("program_info", kg_store.program_info(name))
        except Exception as e:
            return ToolResult("program_info", None, ok=False, error=str(e))

    def _faculty_by_area(args: dict) -> ToolResult:
        area = (args.get("area") or "").strip()
        if not kg_store or not area:
            return ToolResult("faculty_by_area", [], ok=False, error="no area / no kg")
        try:
            return ToolResult("faculty_by_area", {
                "area": area,
                "faculty": kg_store.faculty_by_area(area),
            })
        except Exception as e:
            return ToolResult("faculty_by_area", None, ok=False, error=str(e))

    def _labs_by_area(args: dict) -> ToolResult:
        area = (args.get("area") or "").strip()
        if not kg_store or not area:
            return ToolResult("labs_by_area", [], ok=False, error="no area / no kg")
        try:
            return ToolResult("labs_by_area", {
                "area": area,
                "labs": kg_store.labs_by_area(area),
            })
        except Exception as e:
            return ToolResult("labs_by_area", None, ok=False, error=str(e))

    tools = [
        Tool(
            name="retrieve",
            description=(
                "Hybrid (dense+sparse+rerank) retrieval over UB CSE website. "
                "Use for any question needing grounded content."
            ),
            func=_retrieve,
            schema={"query": "string (required)", "k": "int (default 8)"},
        ),
        Tool(
            name="course_prereqs",
            description="Prerequisites of a CSE course, looked up in the knowledge graph.",
            func=_course_prereqs,
            schema={"code": "string like 'CSE 574'"},
        ),
        Tool(
            name="course_faculty",
            description="Faculty teaching/associated with a CSE course, from the KG.",
            func=_course_faculty,
            schema={"code": "string like 'CSE 574'"},
        ),
        Tool(
            name="related_labs",
            description=(
                "Given a CSE course, suggest related research labs and faculty via KG traversal."
            ),
            func=_related_labs,
            schema={"code": "string like 'CSE 574'"},
        ),
        Tool(
            name="faculty_info",
            description="Look up a faculty member by name in the KG; returns email, office, labs, courses.",
            func=_faculty_info,
            schema={"name": "exact faculty name"},
        ),
        Tool(
            name="lab_info",
            description="Look up a research lab by name in the KG; returns area, url, members.",
            func=_lab_info,
            schema={"name": "exact lab name"},
        ),
        Tool(
            name="program_info",
            description="Look up a program by name in the KG; returns level, url, course count.",
            func=_program_info,
            schema={"name": "exact program name"},
        ),
        Tool(
            name="faculty_by_area",
            description="Find faculty whose lab works in a given research area (e.g. 'machine learning').",
            func=_faculty_by_area,
            schema={"area": "research area keyword"},
        ),
        Tool(
            name="labs_by_area",
            description="Find research labs in a given area (e.g. 'computer vision').",
            func=_labs_by_area,
            schema={"area": "research area keyword"},
        ),
    ]
    return {t.name: t for t in tools}


_CODE = re.compile(r"\b(CSE|EAS|MTH|EE)\s?-?\s?(\d{3}[A-Z]?)\b", re.IGNORECASE)


def _norm_code(s: str) -> str:
    m = _CODE.search(s or "")
    if not m:
        return ""
    return f"{m.group(1).upper()} {m.group(2).upper()}"
