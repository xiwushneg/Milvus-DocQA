"""查询规划：brain 结构化意图分类 / 子查询分解 + 确定性过滤与 HyDE。

设计要点：
- 一次 brain 调用同时产出 intent、sub_queries、needs_table、needs_calculation；
- JSON 解析失败重试 1 次，仍失败回退规则分析器，不中断服务；
- source/page 只允许来自问题中的显式线索，模型臆造的过滤条件会被丢弃；
- 所有 LLM 调用以 llm_traces 返回，由 Runtime 统一记入 trace 与成本分账。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.agentic.state import Intent, SubQuery

PAGE_RE = re.compile(r"第\s*(\d+)\s*页")
DOC_NAME_RE = re.compile(
    r"([\w\u4e00-\u9fff][\w\u4e00-\u9fff.\-]{0,80}\."
    r"(?:pdf|docx|doc|md|txt|pptx|xlsx))",
    re.IGNORECASE,
)

CHITCHAT_RE = re.compile(
    r"^(你好|您好|hi|hello|嗨|谢谢|多谢|再见|你是谁|你能做什么|介绍一下你自己)[!！。?？\s]*$",
    re.IGNORECASE,
)

ANALYZE_PROMPT = """你是 Agentic RAG 的查询规划器。请判断用户问题意图，并拆解为可独立检索的子查询。
只输出一个 JSON 对象，不要 Markdown 代码块、不要解释。格式：
{
  "intent": "fact | comparison | summary | multihop | chitchat | out_of_scope",
  "sub_queries": [
    {"text": "可直接用于检索的独立问题", "channels": ["dense", "bm25"],
     "source": null, "page": null, "purpose": "这一跳要查什么"}
  ],
  "needs_table": false,
  "needs_calculation": false
}
规则：
- fact：单一事实，1 个子查询；comparison：按实体拆 2-3 个；summary：1-2 个；
  multihop：按推理跳数拆 2-3 个；chitchat / out_of_scope：sub_queries 为空数组。
- source 只填用户明确提到的文档名，未提到填 null；page 只填用户明确说出的页码，否则 null。
- channels 默认 ["dense", "bm25"]；专有名词、编号、公式、代码可用 ["bm25"]。
- 不要臆造文档名、页码，也不要回答用户问题。"""

HYDE_PROMPT = """你是领域专家。请针对用户问题写一段 2-4 句的假设性答案，
尽量包含可能出现在目标文档中的术语、实体、数字与表述方式。
只输出假设答案正文，不要解释、不要前缀、不要引号。"""

_INTENT_ALIASES = {
    "fact": Intent.FACT,
    "事实": Intent.FACT,
    "comparison": Intent.COMPARISON,
    "对比": Intent.COMPARISON,
    "比较": Intent.COMPARISON,
    "summary": Intent.SUMMARY,
    "汇总": Intent.SUMMARY,
    "总结": Intent.SUMMARY,
    "multihop": Intent.MULTIHOP,
    "多跳": Intent.MULTIHOP,
    "chitchat": Intent.CHITCHAT,
    "闲聊": Intent.CHITCHAT,
    "out_of_scope": Intent.OUT_OF_SCOPE,
    "越界": Intent.OUT_OF_SCOPE,
    "无关": Intent.OUT_OF_SCOPE,
}


@dataclass
class Analysis:
    intent: Intent
    plan: list[SubQuery] = field(default_factory=list)
    needs_table: bool = False
    needs_calculation: bool = False
    source: str = "rules"  # rules | rules_fast | brain
    degraded: bool = False
    llm_traces: list[dict] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "sub_queries": [sq.to_dict() for sq in self.plan],
            "needs_table": self.needs_table,
            "needs_calculation": self.needs_calculation,
            "source": self.source,
            "degraded": self.degraded,
            "error": self.error,
        }


@dataclass
class DetectedFilter:
    source: str | None = None
    page: int | None = None


def _match_source(
    candidate: str, sources: list[str] | None, question: str = ""
) -> str | None:
    """把问题中的文档名映射到已入库 source；匹配不到返回 None。"""
    cand = (candidate or "").strip().strip("《》“”\"'")
    if not cand:
        return None
    if sources is None:
        # 未知语料清单时，只接受问题里显式出现的文件名，防止模型臆造过滤
        q = (question or "").lower()
        stem = Path(cand).stem.lower()
        if q and (cand.lower() in q or (len(stem) >= 2 and stem in q)):
            return cand
        return None
    for src in sources:
        if cand == src or Path(cand).name == Path(src).name:
            return src
    cand_stem = Path(cand).stem.lower()
    for src in sources:
        if cand_stem and cand_stem == Path(src).stem.lower():
            return src
    return None


def extract_filters(question: str, sources: list[str] | None = None) -> DetectedFilter:
    """从问题中提取显式 source/page 约束（不含模型推断）。"""
    q = question or ""
    detected = DetectedFilter()
    match = PAGE_RE.search(q)
    if match:
        detected.page = int(match.group(1))

    for m in DOC_NAME_RE.finditer(q):
        source = _match_source(m.group(1), sources, q)
        if source:
            detected.source = source
            break

    if detected.source is None and sources:
        for src in sources:
            stem = Path(src).stem
            if len(stem) >= 2 and (src in q or stem in q):
                detected.source = src
                break
    return detected


def _normalize_channels(value) -> tuple[str, ...]:
    allowed = ("dense", "bm25")
    if not isinstance(value, list):
        return allowed
    chosen = {str(v).strip().lower() for v in value}
    out = tuple(ch for ch in allowed if ch in chosen)
    return out or allowed


def _allowed_pages(question: str) -> set[int]:
    return {int(m) for m in PAGE_RE.findall(question or "")}


def _sanitize_plan(
    raw_plan,
    *,
    question: str,
    sources: list[str] | None,
    max_subqueries: int = 4,
) -> list[SubQuery]:
    if not isinstance(raw_plan, list):
        return []
    detected = extract_filters(question, sources)
    allowed_pages = _allowed_pages(question)
    plan: list[SubQuery] = []
    for item in raw_plan[:max_subqueries]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        source = item.get("source")
        if not isinstance(source, str) or not source.strip():
            source = None
        else:
            source = _match_source(source, sources, question)
        page = item.get("page")
        if isinstance(page, bool):
            page = None
        elif isinstance(page, str) and page.isdigit():
            page = int(page)
        elif not isinstance(page, int):
            page = None
        if page is not None and page not in allowed_pages:
            page = None  # 只接受问题中明确出现的页码
        plan.append(
            SubQuery(
                text=text[:300],
                channels=_normalize_channels(item.get("channels")),
                source=source or detected.source,
                page=page if page is not None else detected.page,
                purpose=str(item.get("purpose") or "")[:80],
            )
        )
    return plan


def extract_json(text: str) -> dict | None:
    body = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", body, re.DOTALL | re.IGNORECASE)
    if fenced:
        body = fenced.group(1).strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(body[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _coerce_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "是"}
    return bool(value)


class RuleAnalyzer:
    """无模型降级路径：只区分闲聊/越界/事实检索，并应用显式过滤。"""

    def analyze(
        self,
        question: str,
        *,
        sources: list[str] | None = None,
        focus_source: str | None = None,
    ) -> Analysis:
        q = (question or "").strip()
        if not q:
            return Analysis(Intent.OUT_OF_SCOPE, source="rules_fast")
        if CHITCHAT_RE.match(q):
            return Analysis(Intent.CHITCHAT, source="rules_fast")
        detected = extract_filters(q, sources)
        return Analysis(
            intent=Intent.FACT,
            plan=[
                SubQuery(
                    text=q,
                    source=detected.source or focus_source,
                    page=detected.page,
                    purpose="主查询",
                )
            ],
        )


class BrainAnalyzer:
    """brain 模型结构化分析；失败重试后回退 RuleAnalyzer。"""

    def __init__(
        self,
        brain,
        *,
        fallback: RuleAnalyzer | None = None,
        max_retries: int = 1,
        max_subqueries: int = 4,
    ):
        self.brain = brain
        self.fallback = fallback or RuleAnalyzer()
        self.max_retries = max(0, int(max_retries))
        self.max_subqueries = max(1, int(max_subqueries))

    def _messages(self, question: str, sources: list[str] | None) -> list[dict]:
        hint = ""
        if sources:
            listed = "\n".join(f"- {s}" for s in sources[:30])
            hint = f"\n可选文档名（source 只能从这里选择）：\n{listed}"
        return [
            {"role": "system", "content": ANALYZE_PROMPT + hint},
            {"role": "user", "content": question},
        ]

    def _build(
        self,
        data: dict,
        *,
        question: str,
        sources: list[str] | None,
        traces: list[dict],
    ) -> Analysis | None:
        raw_intent = str(data.get("intent") or "").strip().lower()
        intent = _INTENT_ALIASES.get(raw_intent)
        if intent is None:
            return None
        plan = _sanitize_plan(
            data.get("sub_queries"),
            question=question,
            sources=sources,
            max_subqueries=self.max_subqueries,
        )
        if intent in (Intent.CHITCHAT, Intent.OUT_OF_SCOPE):
            plan = []
        elif not plan:
            detected = extract_filters(question, sources)
            plan = [
                SubQuery(
                    text=question[:300],
                    source=detected.source,
                    page=detected.page,
                    purpose="主查询",
                )
            ]
        return Analysis(
            intent=intent,
            plan=plan,
            needs_table=_coerce_bool(data.get("needs_table")),
            needs_calculation=_coerce_bool(data.get("needs_calculation")),
            source="brain",
            llm_traces=traces,
        )

    def analyze(
        self,
        question: str,
        *,
        sources: list[str] | None = None,
        focus_source: str | None = None,
    ) -> Analysis:
        q = (question or "").strip()
        # 空问题与明显闲聊走规则快路径，不消耗 brain 调用
        fast = self.fallback.analyze(q, sources=sources, focus_source=focus_source)
        if not q or fast.intent in (Intent.OUT_OF_SCOPE, Intent.CHITCHAT):
            return fast

        messages = self._messages(q, sources)
        traces: list[dict] = []
        last_error = ""
        for _ in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                raw = self.brain.chat(messages)
            except Exception as exc:  # 网络/服务错误同样进入降级路径
                last_error = f"llm_error: {exc}"
                continue
            latency_ms = int((time.perf_counter() - t0) * 1000)
            traces.append(
                {"messages": messages, "completion": raw or "", "latency_ms": latency_ms}
            )
            data = extract_json(raw or "")
            if data is None:
                last_error = "json_parse_error"
                continue
            analysis = self._build(
                data, question=q, sources=sources, traces=traces
            )
            if analysis is not None:
                return analysis
            last_error = "schema_error"

        degraded = self.fallback.analyze(q, sources=sources, focus_source=focus_source)
        degraded.degraded = True
        degraded.source = "rules"
        degraded.llm_traces = traces
        degraded.error = last_error
        return degraded


def build_hyde_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": HYDE_PROMPT},
        {"role": "user", "content": question},
    ]


class HydeGenerator:
    """生成假设答案，仅用于 dense 通道；调用失败由 Runtime 降级。"""

    def __init__(self, brain, max_chars: int = 400):
        self.brain = brain
        self.max_chars = max(50, int(max_chars))

    def generate(self, question: str) -> tuple[str, dict]:
        messages = build_hyde_messages(question)
        t0 = time.perf_counter()
        raw = self.brain.chat(messages)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        text = (raw or "").strip().strip('"“”').strip()[: self.max_chars]
        return text, {"messages": messages, "completion": raw or "", "latency_ms": latency_ms}


def build_analyzer(cfg: dict, brain) -> RuleAnalyzer | BrainAnalyzer:
    """按配置构建分析器：agent.analyze.mode = brain | rules。"""
    acfg = cfg.get("agent", {}) or {}
    scfg = acfg.get("analyze", {}) or {}
    if scfg.get("mode", "brain") == "rules" or brain is None:
        return RuleAnalyzer()
    return BrainAnalyzer(
        brain,
        max_retries=int(scfg.get("max_retries", 1)),
        max_subqueries=int(acfg.get("max_subqueries_per_round", 3)),
    )


__all__ = [
    "Analysis",
    "BrainAnalyzer",
    "DetectedFilter",
    "HydeGenerator",
    "RuleAnalyzer",
    "build_analyzer",
    "build_hyde_messages",
    "extract_filters",
    "extract_json",
]
