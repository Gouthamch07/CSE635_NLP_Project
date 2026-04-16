from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


EdgeType = Literal[
    "PREREQUISITE_OF",
    "TAUGHT_BY",
    "PART_OF_PROGRAM",
    "MEMBER_OF_LAB",
    "RESEARCHES",
    "MENTIONS",
]


@dataclass
class Course:
    code: str                 # e.g. CSE 574
    title: str = ""
    description: str = ""
    credits: str = ""
    url: str = ""


@dataclass
class Faculty:
    name: str
    title: str = ""
    email: str = ""
    office: str = ""
    url: str = ""


@dataclass
class Program:
    name: str                 # BS, MS, PhD, Online MS
    level: str = ""           # undergraduate/graduate
    url: str = ""


@dataclass
class Lab:
    name: str
    area: str = ""            # e.g. AI/ML, Systems
    url: str = ""


@dataclass
class Edge:
    src_type: str
    src_key: str              # e.g. course code, faculty name
    dst_type: str
    dst_key: str
    rel: EdgeType
    props: dict = field(default_factory=dict)
