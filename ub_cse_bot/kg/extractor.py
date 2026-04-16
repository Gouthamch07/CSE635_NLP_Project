from __future__ import annotations

import re
from dataclasses import dataclass

from ..utils.logging import get_logger
from .schema import Course, Edge, Faculty, Lab, Program

log = get_logger(__name__)


# e.g. "CSE 574", "CSE574", "EAS 230"
_COURSE_RE = re.compile(r"\b(CSE|EAS|MTH|EE)\s?-?\s?(\d{3}[A-Z]?)\b", re.IGNORECASE)
_PREREQ_RE = re.compile(r"prerequisite[s]?\s*[:\-]\s*([^.]+?)(?:\.|\n)", re.IGNORECASE)
_CREDIT_RE = re.compile(r"(\d+)\s*credit[s]?", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.edu")

_PROGRAM_PATTERNS = {
    "BS in Computer Science": "undergraduate",
    "BS in Computer Engineering": "undergraduate",
    "MS in Computer Science and Engineering": "graduate",
    "PhD in Computer Science and Engineering": "graduate",
    "Online MS": "graduate",
}

_LAB_KEYWORDS = [
    ("AI and Machine Learning", "AI/ML"),
    ("Data Mining", "AI/ML"),
    ("Computer Vision", "AI/ML"),
    ("Natural Language Processing", "AI/ML"),
    ("Systems and Networking", "Systems"),
    ("Computer Architecture", "Systems"),
    ("Cryptography", "Security"),
    ("Security", "Security"),
    ("HCI", "HCI"),
    ("Theory", "Theory"),
    ("Algorithms", "Theory"),
    ("Bioinformatics", "Bio"),
    ("Robotics", "Robotics"),
]


@dataclass
class ExtractionResult:
    courses: dict[str, Course]
    faculty: dict[str, Faculty]
    programs: dict[str, Program]
    labs: dict[str, Lab]
    edges: list[Edge]


class EntityExtractor:
    """Regex + heuristic extractor. LLM-assisted refinement lives in `refine_with_llm`."""

    def __init__(self) -> None:
        self.courses: dict[str, Course] = {}
        self.faculty: dict[str, Faculty] = {}
        self.programs: dict[str, Program] = {}
        self.labs: dict[str, Lab] = {}
        self.edges: list[Edge] = []

    # ---------- normalization ----------
    @staticmethod
    def normalize_course(code: str) -> str:
        m = _COURSE_RE.search(code)
        if not m:
            return code.upper().replace(" ", "")
        return f"{m.group(1).upper()} {m.group(2).upper()}"

    # ---------- extraction entry ----------
    def ingest(self, doc: dict) -> None:
        url = doc.get("url", "")
        text = doc.get("text", "")
        title = doc.get("title", "")

        self._extract_courses(text, url)
        self._extract_faculty(text, url, title)
        self._extract_programs(text, url, title)
        self._extract_labs(text, url, title)
        self._extract_prereqs(text)
        self._extract_teaches(text, url)

    # ---------- specific extractors ----------
    def _extract_courses(self, text: str, url: str) -> None:
        for m in _COURSE_RE.finditer(text):
            code = self.normalize_course(m.group(0))
            if code not in self.courses:
                self.courses[code] = Course(code=code, url=url)
            credit_m = _CREDIT_RE.search(text[max(0, m.start() - 60): m.end() + 60])
            if credit_m:
                self.courses[code].credits = self.courses[code].credits or credit_m.group(0)

    def _extract_faculty(self, text: str, url: str, title: str) -> None:
        if "faculty" not in url.lower() and "people" not in url.lower():
            return
        for email in _EMAIL_RE.findall(text):
            local = email.split("@", 1)[0]
            name = local.replace(".", " ").replace("_", " ").title()
            if name not in self.faculty:
                self.faculty[name] = Faculty(name=name, email=email, url=url)
            else:
                self.faculty[name].email = self.faculty[name].email or email
        if title and any(t in title for t in ["Ph.D.", "Professor", "Assistant", "Associate"]):
            name = re.split(r"[|,-]", title)[0].strip()
            if name and name not in self.faculty:
                self.faculty[name] = Faculty(name=name, url=url)

    def _extract_programs(self, text: str, url: str, title: str) -> None:
        hay = f"{title}\n{text[:2000]}"
        for pname, level in _PROGRAM_PATTERNS.items():
            if pname.lower() in hay.lower() and pname not in self.programs:
                self.programs[pname] = Program(name=pname, level=level, url=url)

    def _extract_labs(self, text: str, url: str, title: str) -> None:
        hay = f"{title}\n{text[:3000]}"
        for kw, area in _LAB_KEYWORDS:
            if kw.lower() in hay.lower():
                key = kw
                if key not in self.labs:
                    self.labs[key] = Lab(name=kw, area=area, url=url)

    def _extract_prereqs(self, text: str) -> None:
        for pm in _PREREQ_RE.finditer(text):
            chunk = pm.group(1)
            codes = [self.normalize_course(c.group(0)) for c in _COURSE_RE.finditer(chunk)]
            # find nearest preceding course code as the subject
            head = text[: pm.start()]
            subj_match = None
            for m in _COURSE_RE.finditer(head):
                subj_match = m
            if not subj_match or not codes:
                continue
            subject = self.normalize_course(subj_match.group(0))
            for pre in codes:
                if pre == subject:
                    continue
                self.edges.append(
                    Edge(
                        src_type="Course", src_key=pre,
                        dst_type="Course", dst_key=subject,
                        rel="PREREQUISITE_OF",
                    )
                )

    def _extract_teaches(self, text: str, url: str) -> None:
        # Heuristic: "Instructor: Dr. X" or "Taught by X"
        pat = re.compile(
            r"(?:Instructor|Taught by|Professor)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
        )
        for m in pat.finditer(text):
            name = m.group(1).strip()
            near = text[max(0, m.start() - 200): m.start() + 200]
            for c in _COURSE_RE.finditer(near):
                code = self.normalize_course(c.group(0))
                self.faculty.setdefault(name, Faculty(name=name, url=url))
                self.edges.append(
                    Edge(
                        src_type="Faculty", src_key=name,
                        dst_type="Course", dst_key=code,
                        rel="TAUGHT_BY",
                    )
                )

    # ---------- result ----------
    def result(self) -> ExtractionResult:
        # dedupe edges
        seen: set[tuple] = set()
        dedup: list[Edge] = []
        for e in self.edges:
            k = (e.src_type, e.src_key, e.dst_type, e.dst_key, e.rel)
            if k in seen:
                continue
            seen.add(k)
            dedup.append(e)
        self.edges = dedup
        log.info(
            "Extracted: %d courses, %d faculty, %d programs, %d labs, %d edges",
            len(self.courses), len(self.faculty), len(self.programs),
            len(self.labs), len(self.edges),
        )
        return ExtractionResult(
            courses=self.courses, faculty=self.faculty,
            programs=self.programs, labs=self.labs, edges=self.edges,
        )
