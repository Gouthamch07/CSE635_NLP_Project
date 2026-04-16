from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Iterator

from ..dialogue.memory import ConversationMemory, PersonalMemory
from ..guardrails.scope import ScopeClassifier, ScopeDecision
from ..kg.neo4j_store import Neo4jStore
from ..llm.vertex_client import LLMMessage, VertexGemini
from ..rag.hybrid import HybridRetriever
from ..utils.logging import get_logger
from .tools import ToolResult, build_tool_registry

log = get_logger(__name__)


SYSTEM_PROMPT = """You are the UB CSE Assistant — a chatbot for the University at Buffalo's
Computer Science and Engineering department. You answer questions about programs
(BS / MS / PhD), courses, faculty, research labs, and departmental policies.

Rules:
- Ground every factual claim in the retrieved snippets. If a claim is not supported,
  say you don't have that info and suggest a UB CSE page to check.
- Cite sources inline as [n] where n is the index of the retrieved snippet you used.
- If the user's question is out of scope, politely redirect to CSE topics.
- Keep small talk brief and friendly, then offer CSE help.
- For follow-ups, resolve pronouns (e.g., "their office hours") against the
  most recent entity in conversation history.

When you need information:
- CALL TOOL <tool_name>{"arg":"value"}</TOOL>
Available tools: retrieve, course_prereqs, course_faculty, related_labs.
After tool results are provided, write the final answer with inline [n] citations
and a short sources list at the end.
"""


@dataclass
class AgentResponse:
    text: str
    scope: ScopeDecision
    tool_calls: list[dict] = field(default_factory=list)
    retrieval_trace: dict | None = None
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    sources: list[dict] = field(default_factory=list)


_TOOL_RE = re.compile(r"CALL\s+TOOL\s+(\w+)\s*(\{.*?\})\s*</?TOOL>?", re.DOTALL)


class UBCSEAgent:
    """Agentic loop: scope-check -> plan tool calls -> ground -> answer.

    We do not rely on Gemini function-calling JSON-schema so the same
    orchestrator runs on any generation backend; parsing a simple
    "CALL TOOL name{json}" envelope works well in practice.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: VertexGemini | None = None,
        kg_store: Neo4jStore | None = None,
        memory: ConversationMemory | None = None,
        personal: PersonalMemory | None = None,
        user_id: str = "anon",
    ) -> None:
        self.retriever = retriever
        self.llm = llm or VertexGemini()
        self.kg_store = kg_store
        self.tools = build_tool_registry(retriever, kg_store)
        self.memory = memory or ConversationMemory()
        self.personal = personal
        self.user_id = user_id
        self.scope = ScopeClassifier(llm=self.llm)

    # ---------- main entry ----------
    def respond(self, user_query: str) -> AgentResponse:
        t0 = time.time()
        scope = self.scope.classify(user_query)

        if scope.label == "out_of_scope":
            ans = scope.redirect
            self.memory.add("user", user_query)
            self.memory.add("model", ans)
            return AgentResponse(text=ans, scope=scope, total_ms=(time.time() - t0) * 1000)

        if scope.label == "small_talk":
            ans = self._small_talk(user_query)
            self.memory.add("user", user_query)
            self.memory.add("model", ans)
            return AgentResponse(text=ans, scope=scope, total_ms=(time.time() - t0) * 1000)

        # Personalized cache hit
        cache_key = _normalize_cache_key(user_query)
        if self.personal and self.personal.enabled_for(self.user_id):
            cached = self.personal.get_cached_answer(self.user_id, cache_key)
            if cached:
                log.info("personal-memory cache hit (%s)", self.user_id)
                self.memory.add("user", user_query)
                self.memory.add("model", cached)
                return AgentResponse(
                    text=cached, scope=scope,
                    total_ms=(time.time() - t0) * 1000,
                    ttft_ms=(time.time() - t0) * 1000,
                )

        # ---- plan + (optional) tool calls
        plan_prompt = self._build_plan_prompt(user_query)
        plan_text = self.llm.generate(plan_prompt, temperature=0.1, max_output_tokens=512)
        tool_calls, tool_results = self._run_tools(plan_text, user_query)

        # ---- final grounded answer
        final_msgs = self._build_answer_prompt(user_query, tool_results)
        ttft_start = time.time()
        answer = self.llm.generate(final_msgs, temperature=0.2, max_output_tokens=1024)
        ttft = (time.time() - ttft_start) * 1000

        # stash trace (first retrieve tool's trace, if any)
        retrieval_trace = None
        sources = []
        for tr in tool_results:
            if tr.name == "retrieve" and tr.ok:
                retrieval_trace = tr.trace
                sources = [
                    {"index": i + 1, "url": h["url"], "title": h["title"], "section": h["section"]}
                    for i, h in enumerate(tr.payload.get("hits", []))
                ]
                break

        self.memory.add("user", user_query)
        self.memory.add("model", answer)

        if self.personal and self.personal.enabled_for(self.user_id):
            self.personal.put_cached_answer(self.user_id, cache_key, answer)
            self._extract_and_save_facts(user_query)

        return AgentResponse(
            text=answer,
            scope=scope,
            tool_calls=[{"name": t.name, "args": t.args} for t in tool_calls],
            retrieval_trace=retrieval_trace,
            sources=sources,
            ttft_ms=ttft,
            total_ms=(time.time() - t0) * 1000,
        )

    # ---------- streaming ----------
    def stream(self, user_query: str) -> Iterator[str]:
        """Yields tokens for UI streaming. Tool calls happen upfront."""
        scope = self.scope.classify(user_query)
        if scope.label == "out_of_scope":
            yield scope.redirect
            return
        if scope.label == "small_talk":
            yield self._small_talk(user_query)
            return
        plan_text = self.llm.generate(self._build_plan_prompt(user_query), temperature=0.1)
        _, tool_results = self._run_tools(plan_text, user_query)
        final_msgs = self._build_answer_prompt(user_query, tool_results)
        yield from self.llm.stream(final_msgs)

    # ---------- internals ----------
    def _build_plan_prompt(self, user_query: str) -> list[LLMMessage]:
        tool_spec = "\n".join(
            f"- {t.name}: {t.description} args={json.dumps(t.schema)}"
            for t in self.tools.values()
        )
        history = "\n".join(f"{t.role}: {t.content}" for t in self.memory.turns())
        personal_facts = ""
        if self.personal and self.personal.enabled_for(self.user_id):
            facts = self.personal.facts(self.user_id)
            if facts:
                personal_facts = "User facts:\n- " + "\n- ".join(facts) + "\n"

        system = SYSTEM_PROMPT + "\n\nTools:\n" + tool_spec
        last_entities = self.memory.last_entities()
        entity_hint = f"\nRecent entities mentioned: {last_entities}" if last_entities else ""

        planner = (
            f"{personal_facts}Conversation so far:\n{history}{entity_hint}\n\n"
            f"User's new question: {user_query}\n\n"
            "Decide which tools to call (0 or more). Output ONLY tool calls in this exact format:\n"
            'CALL TOOL retrieve {"query": "..."} </TOOL>\n'
            "Then stop. Do not write an answer yet."
        )
        return [LLMMessage("system", system), LLMMessage("user", planner)]

    def _run_tools(
        self, plan_text: str, user_query: str
    ) -> tuple[list, list[ToolResult]]:
        from .tools import ToolCall

        calls: list[ToolCall] = []
        for m in _TOOL_RE.finditer(plan_text):
            name = m.group(1).strip()
            try:
                args = json.loads(m.group(2))
            except Exception:
                args = {}
            if name in self.tools:
                calls.append(ToolCall(name=name, args=args))

        # Always fall back to retrieve() if planner chose nothing
        if not calls:
            calls = [ToolCall(name="retrieve", args={"query": user_query})]

        results: list[ToolResult] = []
        for c in calls:
            log.info("tool_call name=%s args=%s", c.name, c.args)
            results.append(self.tools[c.name].func(c.args))
        return calls, results

    def _build_answer_prompt(
        self, user_query: str, tool_results: list[ToolResult]
    ) -> list[LLMMessage]:
        blocks = []
        src_idx = 0
        for tr in tool_results:
            if tr.name == "retrieve" and tr.ok:
                for hit in tr.payload.get("hits", []):
                    src_idx += 1
                    blocks.append(
                        f"[{src_idx}] {hit.get('title','')} — {hit.get('section','')}\n"
                        f"URL: {hit.get('url','')}\n"
                        f"{hit.get('text','')}"
                    )
            else:
                blocks.append(f"TOOL:{tr.name} => {json.dumps(tr.payload, default=str)[:1200]}")

        history = "\n".join(f"{t.role}: {t.content}" for t in self.memory.turns())
        context = "\n\n".join(blocks) if blocks else "(no retrieved context)"
        user = (
            f"Conversation so far:\n{history}\n\n"
            f"New question: {user_query}\n\n"
            f"Grounding context (cite as [n]):\n{context}\n\n"
            "Write the answer now. If the context is insufficient, say so clearly "
            "and suggest a relevant UB CSE page."
        )
        return [LLMMessage("system", SYSTEM_PROMPT), LLMMessage("user", user)]

    def _small_talk(self, q: str) -> str:
        return (
            "Hi! I'm the UB CSE assistant. I can help with CSE programs, courses, "
            "faculty, and research. What would you like to know?"
        )

    def _extract_and_save_facts(self, user_query: str) -> None:
        if not self.personal:
            return
        q = user_query.lower()
        if "i am" in q or "i'm a" in q:
            self.personal.remember_fact(self.user_id, user_query[:200])
        if "my advisor" in q or "my program" in q:
            self.personal.remember_fact(self.user_id, user_query[:200])


def _normalize_cache_key(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())[:256]
