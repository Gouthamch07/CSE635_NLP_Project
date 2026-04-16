from ub_cse_bot.rag.reranker import LexicalReranker, reciprocal_rank_fusion


def test_lexical_boost_exact_match() -> None:
    lex = LexicalReranker()
    cands = [
        {"id": "a", "score": 0.8, "text": "semantic neighbor about ML"},
        {"id": "b", "score": 0.6, "text": "CSE 574 is the course code"},
    ]
    ranked = lex.rerank("CSE 574", cands)
    assert ranked[0]["id"] == "b"


def test_rrf_fusion() -> None:
    # 'b' appears #1 in dense and #1 in sparse -> it should clearly win.
    dense = [{"id": "b"}, {"id": "a"}, {"id": "c"}]
    sparse = [{"id": "b"}, {"id": "c"}, {"id": "a"}]
    fused = reciprocal_rank_fusion(dense, sparse)
    assert fused[0]["id"] == "b"
    assert {c["id"] for c in fused} == {"a", "b", "c"}
