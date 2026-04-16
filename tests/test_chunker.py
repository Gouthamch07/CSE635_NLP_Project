from ub_cse_bot.embeddings.contextual import ContextualChunker


def test_headings_and_chunks() -> None:
    doc = {
        "url": "https://x/ub.html",
        "title": "UB CSE",
        "text": (
            "# Programs\n\n"
            "We offer BS, MS, PhD.\n\n"
            "## MS in CSE\n\n"
            "The MS in Computer Science and Engineering is 30 credits.\n"
            "Prerequisites: CSE 531, CSE 555.\n\n"
            "## Faculty\n\n"
            "See faculty directory.\n"
        ),
    }
    chunks = ContextualChunker(chunk_tokens=40, overlap=5).chunk_doc(doc)
    assert chunks, "should produce chunks"
    assert any("MS in CSE" in c.section for c in chunks)
    # contextualized_text carries breadcrumb/section info
    assert all("[Section]" in c.contextualized_text for c in chunks)


def test_chunker_handles_no_headings() -> None:
    doc = {"url": "https://x", "title": "Flat", "text": "word " * 300}
    chunks = ContextualChunker(chunk_tokens=80, overlap=10).chunk_doc(doc)
    assert len(chunks) >= 2
