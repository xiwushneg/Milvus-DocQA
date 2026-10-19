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


class DocStats(BaseModel):
    total_chunks: int
    sources: dict[str, int]
