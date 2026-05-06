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

    # ---------- health ----------
    def verify_connectivity(self) -> None:
        with self.session() as s:
            s.run("RETURN 1 AS ok").single()

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

    # ---------- entity-index population ----------
    def list_all_courses(self) -> list[dict]:
        q = "MATCH (c:Course) RETURN c.code AS code, c.title AS title, c.url AS url"
        with self.session() as s:
            return [r.data() for r in s.run(q)]

    def list_all_faculty(self) -> list[dict]:
        q = (
            "MATCH (f:Faculty) "
            "RETURN f.name AS name, f.email AS email, f.office AS office, f.url AS url"
        )
        with self.session() as s:
            return [r.data() for r in s.run(q)]

    def list_all_labs(self) -> list[dict]:
        q = "MATCH (l:Lab) RETURN l.name AS name, l.area AS area, l.url AS url"
        with self.session() as s:
            return [r.data() for r in s.run(q)]

    def list_all_programs(self) -> list[dict]:
        q = "MATCH (p:Program) RETURN p.name AS name, p.level AS level, p.url AS url"
        with self.session() as s:
            return [r.data() for r in s.run(q)]

    # ---------- entity-detail lookups ----------
    def faculty_info(self, name: str) -> dict:
        q = """
        MATCH (f:Faculty {name: $name})
        OPTIONAL MATCH (f)-[:MEMBER_OF_LAB]->(l:Lab)
        OPTIONAL MATCH (f)-[:TAUGHT_BY]->(c:Course)
        RETURN f.name AS name, f.email AS email, f.office AS office, f.url AS url,
               collect(DISTINCT {name: l.name, area: l.area, url: l.url}) AS labs,
               collect(DISTINCT {code: c.code, title: c.title}) AS courses
        """
        with self.session() as s:
            rec = s.run(q, name=name).single()
            if not rec:
                return {}
            data = rec.data()
            data["labs"] = [l for l in data.get("labs", []) if l.get("name")]
            data["courses"] = [c for c in data.get("courses", []) if c.get("code")]
            return data

    def lab_info(self, name: str) -> dict:
        q = """
        MATCH (l:Lab {name: $name})
        OPTIONAL MATCH (l)<-[:MEMBER_OF_LAB]-(f:Faculty)
        RETURN l.name AS name, l.area AS area, l.url AS url,
               collect(DISTINCT {name: f.name, email: f.email}) AS members
        """
        with self.session() as s:
            rec = s.run(q, name=name).single()
            if not rec:
                return {}
            data = rec.data()
            data["members"] = [m for m in data.get("members", []) if m.get("name")]
            return data

    def program_info(self, name: str) -> dict:
        q = """
        MATCH (p:Program {name: $name})
        OPTIONAL MATCH (c:Course)-[:PART_OF_PROGRAM]->(p)
        RETURN p.name AS name, p.level AS level, p.url AS url,
               count(DISTINCT c) AS course_count
        """
        with self.session() as s:
            rec = s.run(q, name=name).single()
            return rec.data() if rec else {}

    # ---------- topic / area lookups ----------
    def faculty_by_area(self, area: str) -> list[dict]:
        q = """
        MATCH (f:Faculty)-[:MEMBER_OF_LAB]->(l:Lab)
        WHERE toLower(l.area) CONTAINS toLower($area)
           OR toLower(l.name) CONTAINS toLower($area)
        RETURN DISTINCT f.name AS name, f.email AS email,
               l.name AS lab, l.area AS area
        ORDER BY f.name
        """
        with self.session() as s:
            return [r.data() for r in s.run(q, area=area)]

    def labs_by_area(self, area: str) -> list[dict]:
        q = """
        MATCH (l:Lab)
        WHERE toLower(l.area) CONTAINS toLower($area)
           OR toLower(l.name) CONTAINS toLower($area)
        RETURN l.name AS name, l.area AS area, l.url AS url
        ORDER BY l.name
        """
        with self.session() as s:
            return [r.data() for r in s.run(q, area=area)]
