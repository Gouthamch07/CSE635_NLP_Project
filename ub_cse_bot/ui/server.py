"""FastAPI chat server for the UB CSE Assistant.

Run: PYTHONPATH=. uvicorn ub_cse_bot.ui.server:app --reload --port 8000
Then open http://localhost:8000
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import get_settings
from ub_cse_bot.agent import UBCSEAgent
from ub_cse_bot.dialogue import ConversationMemory, PersonalMemory
from ub_cse_bot.rag.hybrid import HybridRetriever
from ub_cse_bot.utils.logging import get_logger

STATIC_DIR = Path(__file__).parent / "static"
log = get_logger(__name__)

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
    latency_trace: dict[str, float]
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


@app.on_event("startup")
def warm_start() -> None:
    s = get_settings()
    if not s.warm_start_on_startup:
        return
    try:
        log.info("warm-starting UB CSE agent")
        _get_agent().warmup()
        log.info("warm-start complete")
    except Exception as exc:
        log.warning("warm-start skipped: %s", exc)


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
        latency_trace=resp.latency_trace,
        sources=resp.sources or [],
        retrieval_trace=resp.retrieval_trace,
        tool_calls=resp.tool_calls or [],
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    agent = _get_agent()
    agent.user_id = req.user_id

    def _events():
        try:
            for event in agent.stream_events(req.message):
                yield orjson.dumps(event) + b"\n"
        except Exception as exc:
            log.exception("stream chat failed")
            yield orjson.dumps({"type": "error", "message": str(exc)}) + b"\n"

    return StreamingResponse(_events(), media_type="application/x-ndjson")


@app.post("/clear")
def clear() -> dict[str, str]:
    agent = _get_agent()
    agent.memory.clear()
    return {"status": "cleared"}


class PersonalizeRequest(BaseModel):
    user_id: str = "anon"
    enabled: bool


@app.post("/personalize")
def personalize(req: PersonalizeRequest) -> dict[str, Any]:
    agent = _get_agent()
    if req.enabled:
        agent.personal.enable(req.user_id)
    else:
        agent.personal.disable(req.user_id)
    return {"user_id": req.user_id, "enabled": agent.personal.enabled_for(req.user_id)}
