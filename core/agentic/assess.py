"""充分性评估：可解释规则特征 + brain 逐子查询判定。

Score = w1*claim_support + w2*coverage + w3*top_score + w4*diversity
- claim_support：brain 对每个子查询判定 0 / 0.5 / 1 后取均值；
- coverage：子查询内容词在证据中的覆盖率（规则，可复算）；
- top_score：最佳命中分归一化（rerank 0-1 / 本地 sigmoid / RRF 归一）；
- diversity：证据来源数 / 期望来源数（按子查询数，封顶 1）。

brain 调用失败时 claim_support 退化为 coverage，评估仍可继续（degraded）。
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

from core.agentic.analyze import extract_json
from core.agentic.state import Decision, SubQuery, Sufficiency
from core.bm25 import tokenize

ASSESS_PROMPT = """你是 RAG 充分性评估器。给定用户问题、子问题与检索证据，
判断证据是否足以完整回答。逐个评估子问题（顺序与输入一致）：
score 取 1（充分）、0.5（部分）、0（缺失）。
同时列出缺失方面与证据之间的冲突。
只输出一个 JSON 对象，不要 Markdown 代码块、不要解释：
{"support": [{"sub_query": "子问题", "score": 1, "reason": "依据"}],
 "missing_aspects": ["缺失方面"],
 "conflicts": [{"claim": "冲突结论", "evidence_ids": ["E1", "E2"]}]}"""

_CONTENT_TOKEN = re.compile(r"[\w\u4e00-\u9fff]")


@dataclass
class AssessOutcome:
    sufficiency: Sufficiency
    degraded: bool = False
    error: str = ""
    llm_trace: dict | None = None


def _content_tokens(text: str) -> set[str]:
    return {
        t
        for t in tokenize(text or "")
        if len(t) >= 2 and _CONTENT_TOKEN.search(t)
    }


def _coverage_details(
    sub_queries: list[SubQuery], contexts: list[dict]
) -> tuple[float, list[str]]:
    """返回平均覆盖率与覆盖率不足 0.5 的子问题文本。"""
    if not sub_queries:
        return 0.0, []
    corpus = " ".join(str(c.get("text", "")) for c in contexts).lower()
    corpus_tokens = _content_tokens(corpus)
    ratios: list[float] = []
    weak: list[str] = []
    for sq in sub_queries:
        tokens = _content_tokens(sq.text)
        if not tokens:
            ratios.append(0.0)
            weak.append(sq.text)
            continue
        ratio = len(tokens & corpus_tokens) / len(tokens)
        ratios.append(ratio)
        if ratio < 0.5:
            weak.append(sq.text)
    return sum(ratios) / len(ratios), weak


def _normalize_top_score(contexts: list[dict], rerank_mode: str, rrf_k: int) -> float:
    if not contexts:
        return 0.0
    best = contexts[0]
    rerank_score = best.get("rerank_score")
    if rerank_score is not None:
        value = float(rerank_score)
        if rerank_mode == "api":
            return max(0.0, min(1.0, value))
        return 1.0 / (1.0 + math.exp(-value))
    rrf_best = max(float(c.get("rrf_score", 0.0)) for c in contexts)
    max_rrf = 2.0 / (rrf_k + 1.0)  # dense + bm25 同时排第 1 的理论上限
    return max(0.0, min(1.0, rrf_best / max_rrf))


def _diversity(contexts: list[dict], expected: int) -> float:
    if not contexts:
        return 0.0
    sources = {str(c.get("source", "")) for c in contexts if c.get("source")}
    return min(1.0, len(sources) / max(1, expected))


def _coerce_score(value) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number >= 0.75:
        return 1.0
    if number >= 0.25:
        return 0.5
    return 0.0


def _parse_support(data: dict, expected: int) -> tuple[float, list[str], list[dict]]:
    raw = data.get("support")
    scores: list[float] = []
    if isinstance(raw, list):
        for item in raw[:expected]:
            if isinstance(item, dict):
                scores.append(_coerce_score(item.get("score", 0)))
            else:
                scores.append(_coerce_score(item))
    while len(scores) < expected:
        scores.append(0.0)
    support = sum(scores) / max(1, expected)

    missing = [
        str(x)[:120]
        for x in (data.get("missing_aspects") or [])
        if isinstance(x, str) and x.strip()
    ][:3]
    conflicts = []
    for item in (data.get("conflicts") or [])[:3]:
        if not isinstance(item, dict):
            continue
        ids = [
            str(x)
            for x in (item.get("evidence_ids") or [])
            if isinstance(x, str) and x.startswith("E")
        ]
        claim = str(item.get("claim") or "").strip()
        if claim:
            conflicts.append({"claim": claim[:200], "evidence_ids": ids})
    return support, missing, conflicts


class Assessor:
    def __init__(self, brain, cfg: dict, *, rerank_mode: str = "api"):
        acfg = cfg.get("agent", {}) or {}
        scfg = acfg.get("sufficiency", {}) or {}
        self.brain = brain
        self.tau_low = float(scfg.get("tau_low", 0.45))
        self.tau_high = float(scfg.get("tau_high", 0.65))
        weights = scfg.get("weights", {}) or {}
        self.w_claim = float(weights.get("claim_support", 0.45))
        self.w_coverage = float(weights.get("coverage", 0.25))
        self.w_top = float(weights.get("top_score", 0.20))
        self.w_diversity = float(weights.get("diversity", 0.10))

        retrieval_cfg = (cfg.get("retrieval") or {}) or {}
        self.rrf_k = int(retrieval_cfg.get("rrf_k", 60))
        self.rerank_mode = rerank_mode
        self.max_evidence = int(acfg.get("assess_max_evidence", 6))
        self.max_chars = int(acfg.get("assess_max_chars", 300))

    def _messages(
        self, question: str, sub_queries: list[SubQuery], contexts: list[dict]
    ) -> list[dict]:
        sq_lines = "\n".join(
            f"{i}. {sq.text}" for i, sq in enumerate(sub_queries, 1)
        )
        ev_lines = []
        for ctx in contexts[: self.max_evidence]:
            eid = ctx.get("evidence_id", "")
            ev_lines.append(
                f"[{eid}] （{ctx.get('source')} 第{ctx.get('page')}页）"
                f"{str(ctx.get('text', ''))[: self.max_chars]}"
            )
        user = (
            f"【问题】\n{question}\n\n【子问题】\n{sq_lines or '（无）'}"
            f"\n\n【证据】\n" + ("\n\n".join(ev_lines) or "（无）")
        )
        return [
            {"role": "system", "content": ASSESS_PROMPT},
            {"role": "user", "content": user},
        ]

    def assess(
        self,
        *,
        question: str,
        sub_queries: list[SubQuery],
        contexts: list[dict],
        novelty: float,
    ) -> AssessOutcome:
        coverage, weak = _coverage_details(sub_queries, contexts)
        top_score = _normalize_top_score(contexts, self.rerank_mode, self.rrf_k)
        expected = max(1, min(3, len(sub_queries)))
        diversity = _diversity(contexts, expected)

        claim_support = coverage
        missing = weak
        conflicts: list[dict] = []
        degraded = False
        error = ""
        trace: dict | None = None

        if self.brain is not None and sub_queries:
            messages = self._messages(question, sub_queries, contexts)
            t0 = time.perf_counter()
            try:
                raw = self.brain.chat(messages)
                latency_ms = int((time.perf_counter() - t0) * 1000)
                trace = {
                    "messages": messages,
                    "completion": raw or "",
                    "latency_ms": latency_ms,
                }
                data = extract_json(raw or "")
                if data is None:
                    degraded = True
                    error = "json_parse_error"
                else:
                    claim_support, missing, conflicts = _parse_support(
                        data, len(sub_queries)
                    )
            except Exception as exc:
                degraded = True
                error = f"llm_error: {exc}"
        else:
            degraded = True
            error = "no_brain"

        score = (
            self.w_claim * claim_support
            + self.w_coverage * coverage
            + self.w_top * top_score
            + self.w_diversity * diversity
        )
        if score >= self.tau_high:
            decision = Decision.SUFFICIENT
        elif conflicts and score < self.tau_low:
            decision = Decision.CONFLICT
        else:
            decision = Decision.INSUFFICIENT

        return AssessOutcome(
            sufficiency=Sufficiency(
                score=score,
                claim_support=claim_support,
                coverage=coverage,
                top_score=top_score,
                diversity=diversity,
                novelty=novelty,
                missing_aspects=missing,
                conflicts=conflicts,
                decision=decision,
            ),
            degraded=degraded,
            error=error,
            llm_trace=trace,
        )


def build_assessor(cfg: dict, brain, rerank_mode: str = "api") -> Assessor:
    return Assessor(brain, cfg, rerank_mode=rerank_mode)


__all__ = [
    "ASSESS_PROMPT",
    "AssessOutcome",
    "Assessor",
    "build_assessor",
]
