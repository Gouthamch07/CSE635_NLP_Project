from ub_cse_bot.kg.extractor import EntityExtractor


def test_extract_courses_and_prereqs() -> None:
    doc = {
        "url": "https://x",
        "title": "Graduate Courses",
        "text": (
            "CSE 574 Introduction to Machine Learning (3 credits). "
            "Prerequisites: CSE 531, CSE 555.\n"
            "Instructor: Varun Chandola"
        ),
    }
    ex = EntityExtractor()
    ex.ingest(doc)
    r = ex.result()
    assert "CSE 574" in r.courses
    prereq_edges = [e for e in r.edges if e.rel == "PREREQUISITE_OF"]
    assert {e.src_key for e in prereq_edges} >= {"CSE 531", "CSE 555"}
    taught = [e for e in r.edges if e.rel == "TAUGHT_BY"]
    assert any(e.src_key == "Varun Chandola" and e.dst_key == "CSE 574" for e in taught)
