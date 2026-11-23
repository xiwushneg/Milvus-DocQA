"""缺口驱动查询改写：brain 生成下一轮子查询，失败时按缺失方面规则降级。

约束：只产出新的 SubQuery，不改动 Milvus 数据结构；source/page 复用现有
白名单校验；与当前计划重复的查询会被丢弃，配合新颖度停止避免空转。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from core.agentic.analyze import extract_json, extract_filters
from core.agentic.state import SubQuery, Sufficiency

REWRITE_PROMPT = """你是 RAG 查询改写器。当前检索证据不足以完整回答问题，
请针对缺失方面生成 1-3 个新的检索子查询。
要求：
- 不得重复已有子查询，改换角度、补充限定词、英文术语或具体实体；
- channels 默认 ["dense","bm25"]；专有名词/编号/公式/代码可用 ["bm25"]；
- source 只填用户明确提到的文档名，page 只填用户明确说出的页码，否则为 null。
只输出一个 JSON 对象，不要 Markdown 代码块、不要解释：
{"sub_queries": [{"text": "新的检索问题", "channels": ["dense", "bm25"],
  "source": null, "page": null, "purpose": "补充哪个缺口"}],
 "reason": "改写策略的一句话说明"}"""


@dataclass
class RewriteOutcome:
    plan: list[SubQuery]
    source: str = "none"  # brain | rules | none
    reason: str = ""
    error: str = ""
    llm_trace: dict | None = None


def _norm(text: str) -> str:
    return "".join(str(text or "").lower().split())


def _sanitize(raw_plan, question: str, sources: list[str] | None) -> list[SubQuery]:
    """复用 analyze 的过滤校验规则（source/page 只接受显式线索）。"""
    from core.agentic.analyze import _sanitize_plan

    return _sanitize_plan(raw_plan, question=question, sources=sources, max_subqueries=3)


class Rewriter:
    def __init__(self, brain, max_subqueries: int = 3):
        self.brain = brain
        self.max_subqueries = max(1, int(max_subqueries))

    def _messages(
        self,
        question: str,
        plan: list[SubQuery],
        sufficiency: Sufficiency,
        contexts: list[dict],
    ) -> list[dict]:
        sq_lines = "\n".join(f"- {sq.text}" for sq in plan) or "（无）"
        gaps = "\n".join(f"- {x}" for x in sufficiency.missing_aspects) or "（无）"
        ev_lines = []
        for ctx in contexts[:4]:
            ev_lines.append(
                f"[{ctx.get('evidence_id', '')}] {str(ctx.get('text', ''))[:240]}"
            )
        user = (
            f"【问题】\n{question}\n\n【已有子查询】\n{sq_lines}"
            f"\n\n【充分性】score={sufficiency.score:.2f}，"
            f"覆盖不足的子问题：\n{gaps}"
            f"\n\n【现有证据】\n" + ("\n\n".join(ev_lines) or "（无）")
        )
        return [
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": user},
        ]

    def _rule_plan(
        self,
        question: str,
        plan: list[SubQuery],
        sufficiency: Sufficiency,
        sources: list[str] | None,
    ) -> list[SubQuery]:
        base = plan[0] if plan else None
        cands = list(sufficiency.missing_aspects)
        if not cands:
            cands = [f"{question} 的具体细节与数据"]
        detected = extract_filters(question, sources)
        source = (base.source if base else None) or detected.source
        page = (base.page if base else None) or detected.page
        return [
            SubQuery(
                text=text[:300],
                source=source,
                page=page,
                purpose="缺口补充",
            )
            for text in cands[: self.max_subqueries]
        ]

    @staticmethod
    def _dedup(
        plan: list[SubQuery], existing: list[SubQuery]
    ) -> list[SubQuery]:
        seen = {_norm(sq.text) for sq in existing}
        out: list[SubQuery] = []
        for sq in plan:
            key = _norm(sq.text)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(sq)
        return out

    def rewrite(
        self,
        *,
        question: str,
        plan: list[SubQuery],
        sufficiency: Sufficiency,
        contexts: list[dict],
        sources: list[str] | None = None,
    ) -> RewriteOutcome:
        trace: dict | None = None
        error = ""

        if self.brain is not None:
            messages = self._messages(question, plan, sufficiency, contexts)
            t0 = time.perf_counter()
            try:
                raw = self.brain.chat(messages)
                trace = {
                    "messages": messages,
                    "completion": raw or "",
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                }
                data = extract_json(raw or "")
                if data is not None:
                    new_plan = _sanitize(
                        data.get("sub_queries"), question, sources
                    )
                    new_plan = self._dedup(new_plan, plan)[: self.max_subqueries]
                    if new_plan:
                        return RewriteOutcome(
                            plan=new_plan,
                            source="brain",
                            reason=str(data.get("reason") or "")[:200],
                            llm_trace=trace,
                        )
                    error = "empty_or_duplicate_plan"
                else:
                    error = "json_parse_error"
            except Exception as exc:
                error = f"llm_error: {exc}"

        rule_plan = self._dedup(
            self._rule_plan(question, plan, sufficiency, sources), plan
        )
        if rule_plan:
            return RewriteOutcome(
                plan=rule_plan,
                source="rules",
                reason="rule_gap_fill",
                error=error,
                llm_trace=trace,
            )
        return RewriteOutcome(
            plan=[], source="none", reason="", error=error, llm_trace=trace
        )


__all__ = ["REWRITE_PROMPT", "RewriteOutcome", "Rewriter"]
