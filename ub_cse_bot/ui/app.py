"""Streamlit chat UI for the UB CSE Assistant.

Run: streamlit run ub_cse_bot/ui/app.py
"""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from config import get_settings
from ub_cse_bot.agent import UBCSEAgent
from ub_cse_bot.dialogue import ConversationMemory, PersonalMemory
from ub_cse_bot.rag.hybrid import HybridRetriever

st.set_page_config(page_title="UB CSE Assistant", layout="wide")


@st.cache_resource
def _load_agent(user_id: str) -> UBCSEAgent:
    s = get_settings()
    bm25_path = s.data_dir / "processed" / "bm25.pkl"
    retriever = HybridRetriever.from_disk(bm25_path)
    personal = PersonalMemory(Path(s.memory_store_path))
    memory = ConversationMemory(max_turns=12)
    return UBCSEAgent(
        retriever=retriever,
        memory=memory,
        personal=personal,
        user_id=user_id,
    )


def _init_state() -> None:
    if "user_id" not in st.session_state:
        st.session_state.user_id = "anon"
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{role, content, meta}]
    if "personalized" not in st.session_state:
        st.session_state.personalized = False


def _sidebar(agent: UBCSEAgent) -> None:
    st.sidebar.title("UB CSE Assistant")
    st.sidebar.caption("Ask about CSE programs, courses, faculty, and research.")
    st.sidebar.markdown("---")

    # Personalization toggle (bonus)
    st.sidebar.subheader("Personalization")
    on = st.sidebar.toggle(
        "Remember me to speed up answers",
        value=st.session_state.personalized,
        help="If on, the bot keeps notes about you (program, interests) and caches "
             "answers locally to lower latency on repeat queries. You can turn this off any time.",
    )
    if on and not st.session_state.personalized:
        agent.personal.enable(st.session_state.user_id)
        st.session_state.personalized = True
    elif not on and st.session_state.personalized:
        agent.personal.disable(st.session_state.user_id)
        st.session_state.personalized = False

    if st.sidebar.button("Clear conversation"):
        st.session_state.messages = []
        agent.memory.clear()
        st.rerun()


def _render_retrieval_panel(resp_meta: dict | None) -> None:
    if not resp_meta:
        return
    trace = resp_meta.get("retrieval_trace")
    sources = resp_meta.get("sources") or []

    with st.expander("Retrieval / Rerank Log", expanded=False):
        st.caption(
            f"scope={resp_meta.get('scope_label')} "
            f"ttft={resp_meta.get('ttft_ms', 0):.0f}ms "
            f"total={resp_meta.get('total_ms', 0):.0f}ms"
        )
        tool_calls = resp_meta.get("tool_calls") or []
        if tool_calls:
            st.markdown("**Tool calls**")
            st.json(tool_calls)
        if trace:
            st.markdown("**Rerank stages** (top-10 per stage)")
            for stage in trace.get("stages", []):
                st.markdown(f"*{stage['stage']}*")
                st.table(
                    [
                        {"id": sid[:12], "score": round(sc, 4)}
                        for sid, sc in stage["scores"][:10]
                    ]
                )
            st.markdown("**Final hits**")
            st.table(
                [
                    {
                        "#": i + 1,
                        "title": h["title"][:50],
                        "section": h.get("section", "")[:40],
                        "score": round(h["score"], 4),
                    }
                    for i, h in enumerate(trace.get("hits", []))
                ]
            )
        if sources:
            st.markdown("**Sources**")
            for src in sources:
                st.markdown(f"[{src['index']}] [{src['title']}]({src['url']})  \n_{src['section']}_")


def main() -> None:
    _init_state()
    agent = _load_agent(st.session_state.user_id)
    _sidebar(agent)

    st.title("UB CSE Assistant")
    st.caption("Grounded answers about the UB Computer Science and Engineering department.")

    # Render history
    for m in st.session_state.messages:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                _render_retrieval_panel(m.get("meta"))

    prompt = st.chat_input("Ask about a course, professor, program…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        t0 = time.time()
        with st.spinner("Retrieving and grounding…"):
            resp = agent.respond(prompt)
        st.markdown(resp.text)

        meta = {
            "scope_label": resp.scope.label,
            "tool_calls": resp.tool_calls,
            "retrieval_trace": resp.retrieval_trace,
            "sources": resp.sources,
            "ttft_ms": resp.ttft_ms,
            "total_ms": resp.total_ms or (time.time() - t0) * 1000,
        }
        _render_retrieval_panel(meta)

    st.session_state.messages.append({
        "role": "assistant", "content": resp.text, "meta": meta,
    })


if __name__ == "__main__":
    main()
