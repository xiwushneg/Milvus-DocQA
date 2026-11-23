"""Agentic RAG 的显式状态、预算与用量定义。

设计要点：
- 状态与节点解耦：节点是闭包/纯函数，状态是唯一数据载体，便于单测与回放；
- 预算与用量挂在状态上，循环终止条件可审计；
- 不依赖任何第三方编排框架。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Intent(str, Enum):
    FACT = "fact"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    MULTIHOP = "multihop"
    CHITCHAT = "chitchat"
    OUT_OF_SCOPE = "out_of_scope"


class Decision(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    CONFLICT = "conflict"


class Status(str, Enum):
    RUNNING = "running"
    ANSWERED = "answered"
    REFUSED = "refused"
    PARTIAL = "partial"
    ERROR = "error"


def new_trace_id() -> str:
    return f"tr-{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"


def build_filter_expr(source: str | None = None, page: int | None = None) -> str:
    """构造 Milvus 标量过滤表达式。

    只允许现有字段 source / page，禁止把任意表达式透传给 Milvus。
    """
    clauses: list[str] = []
    if source is not None:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source 必须是非空字符串")
        escaped = source.replace("\\", "\\\\").replace('"', '\\"')
        clauses.append(f'source == "{escaped}"')
    if page is not None:
        if isinstance(page, bool) or not isinstance(page, int) or page < 0:
            raise ValueError("page 必须是非负整数")
        clauses.append(f"page == {page}")
    return " and ".join(clauses)


@dataclass
class SubQuery:
    """一条可执行的子查询；过滤条件只允许 source/page。"""

    text: str
    channels: tuple[str, ...] = ("dense", "bm25")
    source: str | None = None
    page: int | None = None
    use_hyde: bool = False
    purpose: str = ""
    # HyDE 假设答案：只用于 dense 查询，BM25 仍用原 text
    hyde_text: str = ""

    @property
    def filter_expr(self) -> str:
        return build_filter_expr(self.source, self.page)

    @property
    def dense_query(self) -> str:
        return self.hyde_text or self.text

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "channels": list(self.channels),
            "source": self.source,
            "page": self.page,
            "use_hyde": self.use_hyde,
            "hyde_text": self.hyde_text,
            "purpose": self.purpose,
        }


@dataclass
class Evidence:
    evidence_id: str
    chunk_id: int
    text: str
    source: str
    page: int
    chunk_idx: int
    chunk_type: str = "prose"
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float | None = None
    channels: list[str] = field(default_factory=list)
    round_found: int = 1
    sub_query: str = ""
    provenance: str = "retrieval"

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page,
            "chunk_idx": self.chunk_idx,
            "chunk_type": self.chunk_type,
            "dense_score": round(self.dense_score, 6),
            "bm25_score": round(self.bm25_score, 6),
            "rrf_score": round(self.rrf_score, 6),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
            "channels": list(self.channels),
            "round_found": self.round_found,
            "sub_query": self.sub_query,
            "provenance": self.provenance,
        }


@dataclass
class Sufficiency:
    score: float = 0.0
    claim_support: float = 0.0
    coverage: float = 0.0
    top_score: float = 0.0
    diversity: float = 0.0
    novelty: float = 0.0
    missing_aspects: list[str] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    decision: Decision = Decision.INSUFFICIENT

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "claim_support": round(self.claim_support, 4),
            "coverage": round(self.coverage, 4),
            "top_score": round(self.top_score, 4),
            "diversity": round(self.diversity, 4),
            "novelty": round(self.novelty, 4),
            "missing_aspects": list(self.missing_aspects),
            "conflicts": list(self.conflicts),
            "decision": self.decision.value,
        }


@dataclass
class RoleUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    estimated: bool = False

    def add(self, prompt_tokens: int, completion_tokens: int, latency_ms: int, estimated: bool) -> None:
        self.calls += 1
        self.prompt_tokens += max(0, int(prompt_tokens))
        self.completion_tokens += max(0, int(completion_tokens))
        self.latency_ms += max(0, int(latency_ms))
        self.estimated = self.estimated or estimated

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "estimated": self.estimated,
        }


@dataclass
class Usage:
    brain: RoleUsage = field(default_factory=RoleUsage)
    answer: RoleUsage = field(default_factory=RoleUsage)
    tool_calls: int = 0

    @property
    def llm_calls(self) -> int:
        return self.brain.calls + self.answer.calls

    @property
    def prompt_tokens(self) -> int:
        return self.brain.prompt_tokens + self.answer.prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self.brain.completion_tokens + self.answer.completion_tokens

    def to_dict(self) -> dict:
        return {
            "brain": self.brain.to_dict(),
            "answer": self.answer.to_dict(),
            "tool_calls": self.tool_calls,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass
class Budget:
    max_rounds: int = 3
    max_llm_calls: int = 8
    max_brain_calls: int = 6
    max_prompt_tokens: int = 12000
    max_completion_tokens: int = 4000
    max_wall_ms: int = 30000
    max_tool_calls: int = 3
    novelty_epsilon: float = 0.10

    @classmethod
    def from_config(cls, acfg: dict[str, Any]) -> "Budget":
        b = acfg.get("budget", {}) or {}
        return cls(
            max_rounds=int(acfg.get("max_rounds", 3)),
            max_llm_calls=int(b.get("max_llm_calls", 8)),
            max_brain_calls=int(b.get("max_brain_calls", 6)),
            max_prompt_tokens=int(b.get("max_prompt_tokens", 12000)),
            max_completion_tokens=int(b.get("max_completion_tokens", 4000)),
            max_wall_ms=int(b.get("max_wall_ms", 30000)),
            max_tool_calls=int(b.get("max_tool_calls", 3)),
            novelty_epsilon=float(b.get("novelty_epsilon", 0.10)),
        )

    def can_continue(self, usage: Usage, started_at: float) -> bool:
        if usage.llm_calls >= self.max_llm_calls:
            return False
        if usage.brain.calls >= self.max_brain_calls:
            return False
        if usage.prompt_tokens >= self.max_prompt_tokens:
            return False
        if usage.completion_tokens >= self.max_completion_tokens:
            return False
        if usage.tool_calls >= self.max_tool_calls:
            return False
        if (time.perf_counter() - started_at) * 1000 >= self.max_wall_ms:
            return False
        return True


@dataclass
class AgentState:
    trace_id: str = field(default_factory=new_trace_id)
    session_id: str = ""
    question: str = ""
    standalone_question: str = ""
    intent: Intent | None = None
    plan: list[SubQuery] = field(default_factory=list)
    needs_table: bool = False
    needs_calculation: bool = False
    round: int = 0
    evidence: dict[int, Evidence] = field(default_factory=dict)
    contexts: list[dict] = field(default_factory=list)
    references: list[dict] = field(default_factory=list)
    sufficiency: Sufficiency | None = None
    tool_results: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    # 本轮新增 evidence / 命中总数，用于新颖度与检索饱和判定
    round_new_ids: set[int] = field(default_factory=set)
    round_hit_count: int = 0
    rewrite_count: int = 0
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    citation_coverage: float = 0.0
    invalid_citations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    status: Status = Status.RUNNING
    stop_reason: str = ""
    error: str = ""
    budget: Budget = field(default_factory=Budget)
    usage: Usage = field(default_factory=Usage)
    trace: list[dict] = field(default_factory=list)
    llm_id: str = "cloud"
    top_k: int | None = None
    history: list[dict] = field(default_factory=list)
    # B4：会话焦点（仅在问题含指代/页码时启用）
    focus_source: str | None = None
    focus_page: int | None = None
    started_at: float = field(default_factory=time.perf_counter)

    @classmethod
    def new(
        cls,
        question: str,
        *,
        session_id: str = "",
        llm_id: str = "cloud",
        top_k: int | None = None,
        history: list[dict] | None = None,
        budget: Budget | None = None,
        trace_id: str | None = None,
    ) -> "AgentState":
        return cls(
            trace_id=trace_id or new_trace_id(),
            session_id=session_id,
            question=question,
            standalone_question=question,
            llm_id=llm_id,
            top_k=top_k,
            history=list(history or []),
            budget=budget or Budget(),
        )
