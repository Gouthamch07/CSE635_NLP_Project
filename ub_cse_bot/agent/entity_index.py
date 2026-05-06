"""Fast in-memory entity index over the Neo4j knowledge graph.

Built once at agent warmup; used at request time to detect course codes,
faculty/lab/program mentions and topic-intent ("who teaches ML?") so the
orchestrator can fan out targeted KG queries without an LLM planner.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..kg.neo4j_store import Neo4jStore

log = get_logger(__name__)

_CODE_RE = re.compile(r"\b(CSE|EAS|MTH|EE)\s?-?\s?(\d{3}[A-Z]?)\b", re.IGNORECASE)

_FACULTY_INTENT_RE = re.compile(
    r"\b(who\s+teach\w*|who\s+research\w*|professor|faculty|researcher|"
    r"specialist|expert|advisor|works\s+on|research\s+in|research\s+on)\b",
    re.IGNORECASE,
)

_LAB_INTENT_RE = re.compile(
    r"\b(lab|labs|research\s+group|research\s+groups|research\s+center)\b",
    re.IGNORECASE,
)

_TOPIC_KEYWORDS: dict[str, str] = {
    "machine learning": "machine learning",
    "deep learning": "deep learning",
    "neural network": "neural",
    "neural networks": "neural",
    "reinforcement learning": "reinforcement",
    "ml": "ml",
    "ai": "ai",
    "artificial intelligence": "artificial intelligence",
    "computer vision": "computer vision",
    "vision": "vision",
    "image processing": "image",
    "cv": "cv",
    "nlp": "nlp",
    "natural language": "natural language",
    "natural language processing": "natural language",
    "language model": "language model",
    "llm": "llm",
    "llms": "llm",
    "hci": "hci",
    "human-computer interaction": "human-computer",
    "human computer interaction": "human computer",
    "security": "security",
    "cybersecurity": "cyber",
    "cryptography": "crypto",
    "operating system": "operating system",
    "operating systems": "operating system",
    "distributed system": "distributed",
    "distributed systems": "distributed",
    "systems": "systems",
    "algorithms": "algorithm",
    "algorithm": "algorithm",
    "theoretical": "theor",
    "theory": "theory",
    "robotics": "robot",
    "robot": "robot",
    "graphics": "graphics",
    "rendering": "graphics",
    "database": "database",
    "databases": "database",
    "data mining": "data mining",
    "data science": "data science",
    "networking": "network",
    "networks": "network",
    "bioinformatics": "bioinformatic",
    "computational biology": "biology",
    "software engineering": "software",
    "computational genomics": "genomic",
    "high performance computing": "high performance",
    "hpc": "hpc",
}


@dataclass
class KGIntent:
    tool: str
    args: dict
    reason: str = ""


@dataclass
class EntityIndex:
    faculty_names: dict[str, str] = field(default_factory=dict)
    lab_names: dict[str, str] = field(default_factory=dict)
    program_names: dict[str, str] = field(default_factory=dict)
    course_codes: set[str] = field(default_factory=set)

    _faculty_re: re.Pattern | None = field(default=None, repr=False)
    _lab_re: re.Pattern | None = field(default=None, repr=False)
    _program_re: re.Pattern | None = field(default=None, repr=False)

    @property
    def loaded(self) -> bool:
        return bool(
            self.faculty_names or self.lab_names
            or self.program_names or self.course_codes
        )

    def build_from_kg(self, kg_store: "Neo4jStore") -> None:
        try:
            faculty = kg_store.list_all_faculty()
            self.faculty_names = {
                f["name"].lower().strip(): f["name"]
                for f in faculty if f.get("name")
            }
            labs = kg_store.list_all_labs()
            self.lab_names = {
                l["name"].lower().strip(): l["name"]
                for l in labs if l.get("name")
            }
            programs = kg_store.list_all_programs()
            self.program_names = {
                p["name"].lower().strip(): p["name"]
                for p in programs if p.get("name")
            }
            courses = kg_store.list_all_courses()
            self.course_codes = {
                c["code"].upper().strip()
                for c in courses if c.get("code")
            }
            self._faculty_re = self._build_pattern(self.faculty_names, min_len=6)
            self._lab_re = self._build_pattern(self.lab_names, min_len=4)
            self._program_re = self._build_pattern(self.program_names, min_len=3)

            log.info(
                "entity_index loaded faculty=%d labs=%d programs=%d courses=%d",
                len(self.faculty_names), len(self.lab_names),
                len(self.program_names), len(self.course_codes),
            )
        except Exception as e:
            log.warning("entity_index build failed: %s", e)

    @staticmethod
    def _build_pattern(names: dict[str, str], min_len: int = 4) -> re.Pattern | None:
        keys = [k for k in names.keys() if len(k) >= min_len]
        if not keys:
            return None
        sorted_keys = sorted(keys, key=len, reverse=True)
        pattern = r"\b(" + "|".join(re.escape(k) for k in sorted_keys) + r")\b"
        return re.compile(pattern, re.IGNORECASE)

    def detect(self, query: str) -> list[KGIntent]:
        if not query:
            return []
        intents: list[KGIntent] = []
        seen: set[tuple[str, str]] = set()

        def _add(tool: str, key: str, args: dict, reason: str) -> None:
            sig = (tool, key.lower())
            if sig in seen:
                return
            seen.add(sig)
            intents.append(KGIntent(tool=tool, args=args, reason=reason))

        # Course codes
        for m in _CODE_RE.finditer(query):
            code = f"{m.group(1).upper()} {m.group(2).upper()}"
            _add("course_prereqs", code, {"code": code}, "course code")
            _add("course_faculty", code, {"code": code}, "course code")
            _add("related_labs", code, {"code": code}, "course code")

        # Faculty names
        if self._faculty_re:
            for m in self._faculty_re.finditer(query):
                key = m.group(0).lower().strip()
                name = self.faculty_names.get(key)
                if name:
                    _add("faculty_info", name, {"name": name}, "faculty name")

        # Lab names
        if self._lab_re:
            for m in self._lab_re.finditer(query):
                key = m.group(0).lower().strip()
                name = self.lab_names.get(key)
                if name:
                    _add("lab_info", name, {"name": name}, "lab name")

        # Program names
        if self._program_re:
            for m in self._program_re.finditer(query):
                key = m.group(0).lower().strip()
                name = self.program_names.get(key)
                if name:
                    _add("program_info", name, {"name": name}, "program name")

        # Topic intent
        q_lower = query.lower()
        topics = self._detect_topics(q_lower)
        if topics:
            wants_faculty = bool(_FACULTY_INTENT_RE.search(query))
            wants_labs = bool(_LAB_INTENT_RE.search(query))
            broad_research = "research" in q_lower
            for topic in topics:
                if wants_faculty:
                    _add("faculty_by_area", topic, {"area": topic}, "topic+faculty")
                if wants_labs:
                    _add("labs_by_area", topic, {"area": topic}, "topic+lab")
                if broad_research and not wants_faculty and not wants_labs:
                    _add("faculty_by_area", topic, {"area": topic}, "topic+research")
                    _add("labs_by_area", topic, {"area": topic}, "topic+research")
        return intents

    @staticmethod
    def _detect_topics(q_lower: str) -> list[str]:
        found: list[str] = []
        matched_spans: list[tuple[int, int]] = []
        sorted_keys = sorted(_TOPIC_KEYWORDS.keys(), key=len, reverse=True)
        for keyword in sorted_keys:
            for m in re.finditer(rf"\b{re.escape(keyword)}\b", q_lower):
                span = (m.start(), m.end())
                if any(s[0] <= span[0] and s[1] >= span[1] for s in matched_spans):
                    continue
                matched_spans.append(span)
                area = _TOPIC_KEYWORDS[keyword]
                if area not in found:
                    found.append(area)
        return found
