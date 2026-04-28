"""FastAPI chat server for the UB CSE Assistant.

Run: PYTHONPATH=. uvicorn ub_cse_bot.ui.server:app --reload --port 8000
Then open http://localhost:8000
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import get_settings
from ub_cse_bot.agent import UBCSEAgent
from ub_cse_bot.dialogue import ConversationMemory, PersonalMemory
from ub_cse_bot.rag.hybrid import HybridRetriever

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="UB CSE Assistant")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str
    user_id: str = "anon"


class ChatResponse(BaseModel):
    text: str
    scope: str
    ttft_ms: float
    total_ms: float
    sources: list[dict[str, Any]]
    retrieval_trace: dict[str, Any] | None
    tool_calls: list[dict[str, Any]]


_agent: UBCSEAgent | None = None


def _get_agent() -> UBCSEAgent:
    global _agent
    if _agent is None:
        s = get_settings()
        retriever = HybridRetriever.from_disk(s.data_dir / "processed" / "bm25.pkl")
        _agent = UBCSEAgent(
            retriever=retriever,
            memory=ConversationMemory(max_turns=12),
            personal=PersonalMemory(Path(s.memory_store_path)),
            user_id="anon",
        )
    return _agent


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    agent = _get_agent()
    agent.user_id = req.user_id
    resp = agent.respond(req.message)
    return ChatResponse(
        text=resp.text,
        scope=resp.scope.label if resp.scope else "in_scope",
        ttft_ms=resp.ttft_ms,
        total_ms=resp.total_ms,
        sources=resp.sources or [],
        retrieval_trace=resp.retrieval_trace,
        tool_calls=resp.tool_calls or [],
    )


@app.post("/clear")
def clear() -> dict[str, str]:
    agent = _get_agent()
    agent.memory.clear()
    return {"status": "cleared"}
