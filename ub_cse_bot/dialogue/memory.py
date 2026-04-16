from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import orjson

from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Turn:
    role: str
    content: str
    ts: float = field(default_factory=time.time)


class ConversationMemory:
    """Short-term rolling window — used on every turn as prompt context.

    Default window is 8 turns (4 user + 4 assistant) which is enough to
    resolve pronouns in follow-ups like 'what are their office hours?'
    """

    def __init__(self, max_turns: int = 8) -> None:
        self._buf: deque[Turn] = deque(maxlen=max_turns)

    def add(self, role: str, content: str) -> None:
        self._buf.append(Turn(role=role, content=content))

    def turns(self) -> list[Turn]:
        return list(self._buf)

    def as_messages(self) -> list[dict]:
        return [{"role": t.role, "content": t.content} for t in self._buf]

    def clear(self) -> None:
        self._buf.clear()

    # Build a context string that resolves follow-ups.
    def last_entities(self) -> str:
        """Extract recently mentioned courses/faculty for follow-up context."""
        import re
        hay = " ".join(t.content for t in self._buf)
        courses = re.findall(r"\bCSE\s?\d{3}[A-Z]?\b", hay, re.IGNORECASE)
        return "; ".join(sorted(set(c.upper() for c in courses)))


class PersonalMemory:
    """Opt-in long-term memory keyed by user_id.

    Stored as JSON on disk; in production this would be a KV store. We only
    persist *salient* facts (user's program, interests, advisor preferences,
    completed courses) — not the full transcript.

    Why it helps latency: we can cache common queries and skip retrieval when
    a prior answer is still fresh.
    """

    def __init__(self, store_path: Path) -> None:
        self.path = store_path
        self.path.mkdir(parents=True, exist_ok=True)

    def _file(self, user_id: str) -> Path:
        safe = "".join(c for c in user_id if c.isalnum() or c in "-_")
        return self.path / f"{safe or 'anon'}.json"

    def enabled_for(self, user_id: str) -> bool:
        return self._file(user_id).exists()

    def enable(self, user_id: str) -> None:
        p = self._file(user_id)
        if not p.exists():
            p.write_bytes(orjson.dumps({"facts": [], "q_cache": {}, "user_id": user_id}))
            log.info("Personal memory enabled for %s", user_id)

    def disable(self, user_id: str) -> None:
        p = self._file(user_id)
        if p.exists():
            p.unlink()
            log.info("Personal memory disabled + wiped for %s", user_id)

    def _load(self, user_id: str) -> dict:
        p = self._file(user_id)
        if not p.exists():
            return {"facts": [], "q_cache": {}, "user_id": user_id}
        return orjson.loads(p.read_bytes())

    def _save(self, user_id: str, data: dict) -> None:
        self._file(user_id).write_bytes(orjson.dumps(data))

    # ---------- fact store ----------
    def remember_fact(self, user_id: str, fact: str) -> None:
        if not self.enabled_for(user_id):
            return
        data = self._load(user_id)
        if fact not in data["facts"]:
            data["facts"].append(fact)
            self._save(user_id, data)

    def facts(self, user_id: str) -> list[str]:
        if not self.enabled_for(user_id):
            return []
        return self._load(user_id).get("facts", [])

    # ---------- query cache (latency win) ----------
    def get_cached_answer(self, user_id: str, query_key: str, ttl: float = 600.0) -> str | None:
        if not self.enabled_for(user_id):
            return None
        data = self._load(user_id)
        entry = data.get("q_cache", {}).get(query_key)
        if not entry:
            return None
        if time.time() - entry["ts"] > ttl:
            return None
        return entry["answer"]

    def put_cached_answer(self, user_id: str, query_key: str, answer: str) -> None:
        if not self.enabled_for(user_id):
            return
        data = self._load(user_id)
        data.setdefault("q_cache", {})[query_key] = {"answer": answer, "ts": time.time()}
        self._save(user_id, data)
