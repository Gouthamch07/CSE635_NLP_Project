from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

from neo4j import GraphDatabase, Driver

from config import get_settings

from ..utils.logging import get_logger
from .schema import Course, Edge, Faculty, Lab, Program

log = get_logger(__name__)


class Neo4jStore:
    def __init__(self) -> None:
        s = get_settings()
        self._driver: Driver = GraphDatabase.driver(
            s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password)
        )
        self._db = s.neo4j_database

    def close(self) -> None:
        self._driver.close()

    @contextmanager
    def session(self):
        with self._driver.session(database=self._db) as sess:
            yield sess

    # ---------- schema ----------
    def ensure_constraints(self) -> None:
        stmts = [
            "CREATE CONSTRAINT course_code IF NOT EXISTS FOR (c:Course) REQUIRE c.code IS UNIQUE",
            "CREATE CONSTRAINT faculty_name IF NOT EXISTS FOR (f:Faculty) REQUIRE f.name IS UNIQUE",
            "CREATE CONSTRAINT program_name IF NOT EXISTS FOR (p:Program) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT lab_name IF NOT EXISTS FOR (l:Lab) REQUIRE l.name IS UNIQUE",
        ]
        with self.session() as s:
            for stmt in stmts:
                s.run(stmt)

    # ---------- upserts ----------
    def upsert_courses(self, courses: Iterable[Course]) -> None:
        q = """
        UNWIND $rows AS row
        MERGE (c:Course {code: row.code})
        SET c.title = coalesce(row.title, c.title),
            c.description = coalesce(row.description, c.description),
            c.credits = coalesce(row.credits, c.credits),
            c.url = coalesce(row.url, c.url)
        """
        with self.session() as s:
            s.run(q, rows=[c.__dict__ for c in courses])

    def upsert_faculty(self, faculty: Iterable[Faculty]) -> None:
        q = """
        UNWIND $rows AS row
        MERGE (f:Faculty {name: row.name})
        SET f.title = coalesce(row.title, f.title),
            f.email = coalesce(row.email, f.email),
            f.office = coalesce(row.office, f.office),
            f.url = coalesce(row.url, f.url)
        """
        with self.session() as s:
            s.run(q, rows=[f.__dict__ for f in faculty])

    def upsert_programs(self, programs: Iterable[Program]) -> None:
        q = """
        UNWIND $rows AS row
        MERGE (p:Program {name: row.name})
        SET p.level = coalesce(row.level, p.level),
            p.url = coalesce(row.url, p.url)
        """
        with self.session() as s:
            s.run(q, rows=[p.__dict__ for p in programs])

    def upsert_labs(self, labs: Iterable[Lab]) -> None:
        q = """
        UNWIND $rows AS row
        MERGE (l:Lab {name: row.name})
        SET l.area = coalesce(row.area, l.area),
            l.url = coalesce(row.url, l.url)
        """
        with self.session() as s:
            s.run(q, rows=[l.__dict__ for l in labs])

    def upsert_edges(self, edges: Iterable[Edge]) -> None:
        by_rel: dict[str, list[dict]] = {}
        for e in edges:
            by_rel.setdefault(e.rel, []).append({
                "src": e.src_key, "dst": e.dst_key, "props": e.props,
                "src_type": e.src_type, "dst_type": e.dst_type,
            })

        # static mapping of src_type/dst_type keys
        key_prop = {"Course": "code", "Faculty": "name", "Program": "name", "Lab": "name"}

        with self.session() as s:
            for rel, rows in by_rel.items():
                # group by src/dst type pair so we can emit a typed MATCH
                groups: dict[tuple[str, str], list[dict]] = {}
                for r in rows:
                    groups.setdefault((r["src_type"], r["dst_type"]), []).append(r)
                for (st, dt), grp in groups.items():
                    q = f"""
                    UNWIND $rows AS row
                    MATCH (a:{st} {{{key_prop[st]}: row.src}})
                    MATCH (b:{dt} {{{key_prop[dt]}: row.dst}})
                    MERGE (a)-[r:{rel}]->(b)
                    SET r += row.props
                    """
                    s.run(q, rows=grp)

    # ---------- queries used by agent tools ----------
    def related_faculty_for_course(self, code: str) -> list[dict]:
        q = """
        MATCH (c:Course {code: $code})<-[:TAUGHT_BY]-(f:Faculty)
        RETURN f.name AS name, f.email AS email, f.url AS url
        """
        with self.session() as s:
            return [r.data() for r in s.run(q, code=code)]

    def related_labs_for_course(self, code: str) -> list[dict]:
        q = """
        MATCH (c:Course {code: $code})<-[:TAUGHT_BY]-(f:Faculty)-[:MEMBER_OF_LAB]->(l:Lab)
        RETURN DISTINCT l.name AS name, l.area AS area, l.url AS url
        """
        with self.session() as s:
            return [r.data() for r in s.run(q, code=code)]

    def prerequisites(self, code: str) -> list[str]:
        q = """
        MATCH (pre:Course)-[:PREREQUISITE_OF]->(c:Course {code: $code})
        RETURN pre.code AS code
        """
        with self.session() as s:
            return [r["code"] for r in s.run(q, code=code)]
