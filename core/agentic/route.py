"""智能路由：按意图、查询特征与轮次决定检索通道 / HyDE / 证据预算。

约束：路由只决定"怎么用现有检索组件"，不改变 Milvus schema，也不修改
`RagPipeline._retrieve`。RRF 融合顺序固定为 dense → bm25，因此"BM25 优先"
通过只走 bm25 通道、dense 作为零命中兜底来实现（消融可复现）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from core.agentic.state import Intent, SubQuery

# 专有名词 / 英文术语 / 引号术语 / 版本号 / 公式 / 代码
_ASCII_TERM = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{1,}")
_VERSION = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")
_QUOTED = re.compile(r"《[^》]{2,40}》|“[^”]{2,40}”|\"[^\"]{2,40}\"")
_FORMULA = re.compile(r"[=Σ∑∫√≤≥±×÷]")
_CODE = re.compile(
    r"```|\b(SELECT|INSERT|UPDATE|DELETE|CREATE|def|class|import|return"
    r"|function|const|let|var)\b|pip\s+install|npm\s+install",
    re.IGNORECASE,
)
_AGGREGATE = re.compile(r"对比|比较|区别|异同|汇总|总计|合计|平均|最大|最小|最多|最少|趋势|占比|列表|列出")
_CALCULATION = re.compile(
    r"计算|换算|等于多少|乘以|除以|加上|减去|[0-9]+\s*[+\-*/×÷]\s*[0-9]+"
)


@dataclass(frozen=True)
class RouteDecision:
    channels: tuple[str, ...] = ("dense", "bm25")
    fallback_channels: tuple[str, ...] = ()
    use_hyde: bool = False
    evidence_bonus: int = 0
    reason: str = "default"
    needs_table: bool = False
    needs_calculation: bool = False

    def to_dict(self) -> dict:
        return {
            "channels": list(self.channels),
            "fallback_channels": list(self.fallback_channels),
            "use_hyde": self.use_hyde,
            "evidence_bonus": self.evidence_bonus,
            "reason": self.reason,
            "needs_table": self.needs_table,
            "needs_calculation": self.needs_calculation,
        }


def query_features(text: str) -> dict[str, bool]:
    """查询特征（确定性规则，便于 trace 与复现）。"""
    body = text or ""
    return {
        "has_term": bool(_ASCII_TERM.search(body) or _QUOTED.search(body)),
        "has_version": bool(_VERSION.search(body)),
        "has_formula": bool(_FORMULA.search(body)),
        "has_code": bool(_CODE.search(body)),
        "wants_aggregate": bool(_AGGREGATE.search(body)),
        "wants_calculation": bool(_CALCULATION.search(body)),
    }


def _keyword_preferred(features: dict[str, bool]) -> bool:
    return (
        features["has_code"]
        or features["has_version"]
        or features["has_formula"]
        or features["has_term"]
    )


def should_use_hyde(round_no: int, hyde_cfg: dict | None) -> bool:
    """HyDE 仅从配置的起始轮次启用（默认第 2 轮起）。"""
    cfg = hyde_cfg or {}
    if not cfg.get("enabled", False):
        return False
    return round_no >= int(cfg.get("start_round", 2))


def route_subquery(
    sub_query: SubQuery,
    *,
    intent: Intent | None,
    round_no: int,
    hyde_cfg: dict | None = None,
    needs_table: bool = False,
    needs_calculation: bool = False,
    summary_evidence_bonus: int = 2,
    keyword_bm25_first: bool = True,
) -> RouteDecision:
    """为单条子查询生成路由决策。"""
    if intent in (Intent.CHITCHAT, Intent.OUT_OF_SCOPE):
        return RouteDecision(channels=(), reason="no_retrieval")

    features = query_features(sub_query.text)
    hyde = should_use_hyde(round_no, hyde_cfg)
    bonus = summary_evidence_bonus if intent == Intent.SUMMARY else 0

    if sub_query.source is not None or sub_query.page is not None:
        return RouteDecision(
            use_hyde=hyde,
            evidence_bonus=bonus,
            reason="filtered",
            needs_table=needs_table,
            needs_calculation=needs_calculation,
        )

    if keyword_bm25_first and _keyword_preferred(features):
        return RouteDecision(
            channels=("bm25",),
            fallback_channels=("dense",),
            use_hyde=hyde,
            reason="keyword_bm25_first",
            needs_table=needs_table or features["wants_aggregate"],
            needs_calculation=needs_calculation or features["wants_calculation"],
        )

    return RouteDecision(
        use_hyde=hyde,
        evidence_bonus=bonus,
        reason="summary" if bonus else "default",
        needs_table=needs_table or features["wants_aggregate"],
        needs_calculation=needs_calculation or features["wants_calculation"],
    )


__all__ = [
    "RouteDecision",
    "query_features",
    "route_subquery",
    "should_use_hyde",
]
