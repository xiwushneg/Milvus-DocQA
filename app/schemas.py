"""API 数据模型（Pydantic v2）。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class HistoryMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$", description="user 或 assistant")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    top_k: int | None = Field(None, ge=1, le=20, description="最终上下文块数，缺省用配置")
    llm: str = Field("cloud", description="生成模型档案 id，见 GET /api/llms")
    history: list[HistoryMessage] = Field([], description="最近对话历史（用于指代消解/查询改写）")


class Reference(BaseModel):
    source: str
    page: int
    chunk_idx: int = -1
    snippet: str
    score: float = 0.0
    channels: list[str] = []


class ChatResponse(BaseModel):
    answer: str
    references: list[Reference]
    timings: dict
    total_ms: int
    llm: str = "cloud"


class IngestItem(BaseModel):
    source: str
    sha256: str
    segments: int
    chunks: int
    split_ms: int = 0
    embed_ms: int = 0
    elapsed_ms: int


class IngestResponse(BaseModel):
    documents: list[IngestItem]
    skipped: list[dict] = Field(default_factory=list)


class DocStats(BaseModel):
    total_chunks: int
    sources: dict[str, int]


# ==================== Agentic RAG（B0/B1） ====================
class AgentChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    session_id: str | None = Field(None, max_length=64, description="会话 id，缺省由服务端生成")
    top_k: int | None = Field(None, ge=1, le=20, description="每条子查询的最终证据数")
    llm: str = Field("cloud", description="生成模型档案 id（brain 固定走 cloud）")
    history: list[HistoryMessage] = Field(
        default_factory=list, description="最近对话历史（用于指代消解）"
    )
    include_trace: bool = Field(False, description="是否在响应中返回完整轨迹")


class AgentReference(Reference):
    evidence_id: str = ""
    chunk_type: str = "prose"


class AgentCitation(BaseModel):
    sentence: str
    evidence_ids: list[str] = Field(default_factory=list)
    valid: bool = True


class AgentMetrics(BaseModel):
    rounds: int = 0
    sub_queries: int = 0
    rewrites: int = 0
    sufficiency_score: float = 0.0
    citation_coverage: float = 0.0
    invalid_citations: int = 0
    confidence: float = 0.0
    tool_calls: int = 0
    brain_llm_calls: int = 0
    brain_latency_ms: int = 0
    answer_llm_calls: int = 0
    answer_latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_cny: float = 0.0
    latency_ms: int = 0
    stop_reason: str = ""


class AgentChatResponse(BaseModel):
    answer: str
    references: list[AgentReference] = Field(default_factory=list)
    citations: list[AgentCitation] = Field(default_factory=list)
    conflicts: list[dict] = Field(default_factory=list)
    tool_results: list[dict] = Field(default_factory=list)
    confidence: float = 0.0
    refused: bool = False
    refusal_reason: str | None = None
    session_id: str
    trace_id: str
    metrics: AgentMetrics
    llm: str = "cloud"
    trace: list[dict] | None = None
