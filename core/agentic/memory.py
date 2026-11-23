"""会话记忆：多轮历史、焦点文档/页码、最近证据，供指代消解与检索聚焦。

- 进程内 ``SessionStore``，按 TTL 过期，最多保留 ``max_turns`` 轮；
- 可选 JSON 持久化（``persist_path``），默认关闭；
- 焦点只从“该文档 / 它 / 第 X 页”等明确指代中启用，避免过度限制新问题。
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

_PAGE_RE = re.compile(r"第\s*(\d{1,4})\s*页")
_ANAPHORA_RE = re.compile(
    r"它|这个|那个|该文档|该文|这份|那份|上一份|上述|该表|这张表|这个表|该页|这一页|那一页"
)


def detect_reference(question: str) -> dict:
    """检测指代与显式页码：返回 ``{"anaphora": bool, "page": int|None}``。"""
    text = question or ""
    page_match = _PAGE_RE.search(text)
    page = int(page_match.group(1)) if page_match else None
    return {
        "anaphora": bool(page is not None or _ANAPHORA_RE.search(text)),
        "page": page,
    }


@dataclass
class SessionTurn:
    question: str
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": list(self.sources),
            "evidence_ids": list(self.evidence_ids),
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionTurn":
        return cls(
            question=str(data.get("question", "")),
            answer=str(data.get("answer", "")),
            sources=[str(s) for s in data.get("sources", [])],
            evidence_ids=[str(e) for e in data.get("evidence_ids", [])],
            ts=float(data.get("ts", time.time())),
        )


@dataclass
class Session:
    session_id: str
    turns: list[SessionTurn] = field(default_factory=list)
    focus_source: str = ""
    focus_page: int | None = None
    last_evidence_ids: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def history(self, max_turns: int = 6) -> list[dict]:
        """转成 ``build_rewrite_messages`` 可用的 user/assistant 消息。"""
        messages: list[dict] = []
        for turn in self.turns[-max(1, max_turns) :]:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer[:500]})
        return messages

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "turns": len(self.turns),
            "focus_source": self.focus_source,
            "focus_page": self.focus_page,
            "last_evidence_ids": list(self.last_evidence_ids),
        }

    def to_dict(self) -> dict:
        return {
            **self.summary(),
            "turns": [t.to_dict() for t in self.turns],
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(
            session_id=str(data.get("session_id", "")),
            turns=[SessionTurn.from_dict(t) for t in data.get("turns", [])],
            focus_source=str(data.get("focus_source", "")),
            focus_page=(
                None if data.get("focus_page") is None else int(data["focus_page"])
            ),
            last_evidence_ids=[str(e) for e in data.get("last_evidence_ids", [])],
            updated_at=float(data.get("updated_at", time.time())),
        )


class SessionStore:
    """线程安全的会话存储；TTL 到期自动清理。"""

    def __init__(
        self,
        *,
        ttl_s: int = 7200,
        max_turns: int = 20,
        persist_path: str | Path | None = None,
    ):
        self.ttl_s = max(1, int(ttl_s))
        self.max_turns = max(1, int(max_turns))
        self.persist_path = Path(persist_path) if persist_path else None
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()
        self._load()

    def get(self, session_id: str) -> Session | None:
        if not session_id:
            return None
        with self._lock:
            self._prune()
            return self._sessions.get(session_id)

    def append(
        self,
        session_id: str,
        question: str,
        answer: str,
        *,
        sources: list[str] | tuple[str, ...] = (),
        evidence_ids: list[str] | tuple[str, ...] = (),
        focus_source: str = "",
        focus_page: int | None = None,
    ) -> Session:
        if not session_id:
            raise ValueError("session_id 不能为空")
        with self._lock:
            self._prune()
            session = self._sessions.get(session_id) or Session(session_id=session_id)
            session.turns.append(
                SessionTurn(
                    question=question,
                    answer=answer,
                    sources=list(dict.fromkeys(sources)),
                    evidence_ids=list(dict.fromkeys(evidence_ids)),
                )
            )
            session.turns = session.turns[-self.max_turns :]
            if focus_source:
                session.focus_source = focus_source
            if focus_page is not None:
                session.focus_page = focus_page
            session.last_evidence_ids = list(dict.fromkeys(evidence_ids))[:8]
            session.updated_at = time.time()
            self._sessions[session_id] = session
            self._save()
            return session

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._save()

    def __len__(self) -> int:
        with self._lock:
            self._prune()
            return len(self._sessions)

    # ---------- 内部 ----------
    def _prune(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, session in self._sessions.items()
            if now - session.updated_at > self.ttl_s
        ]
        for sid in expired:
            self._sessions.pop(sid, None)

    def _load(self) -> None:
        if self.persist_path is None or not self.persist_path.is_file():
            return
        try:
            payload = json.loads(self.persist_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for item in payload.get("sessions", []):
            try:
                session = Session.from_dict(item)
            except (TypeError, ValueError):
                continue
            self._sessions[session.session_id] = session

    def _save(self) -> None:
        if self.persist_path is None:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "sessions": [s.to_dict() for s in self._sessions.values()],
            }
            tmp = self.persist_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.replace(self.persist_path)
        except OSError:
            # 持久化失败不影响会话内记忆
            pass


__all__ = ["Session", "SessionStore", "SessionTurn", "detect_reference"]
