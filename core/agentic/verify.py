"""答案校验：逐句引用、非法引用检出、冲突感知置信度与低置信拒答。

置信度口径（可在设计文档 §5.6 复核）：
    confidence = 0.40*sufficiency + 0.30*claim_support
               + 0.20*citation_coverage + 0.10*(1 if 无冲突 else 0)
- citation_coverage：含至少一个有效引用的事实句占比；
- claim_support：brain 逐句校验（可选，失败退化为 coverage，不阻塞回答）；
- coverage 低于 ``min_coverage`` 时置信度强制低于 ``tau``，进入拒答。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.agentic.analyze import extract_json
from core.agentic.compose import ok_tool_results
from core.agentic.state import Sufficiency

VERIFY_PROMPT = """你是引用校验器。给定答案句子及其引用证据，逐句判断证据是否支持该句。
只输出一个 JSON 对象，不要 Markdown 代码块、不要解释：
{"claims": [{"index": 1, "supported": true, "reason": "依据"}]}
index 与输入句子编号一致；证据不足以支持时 supported 取 false。"""

_CITE_RE = re.compile(r"\[(E\d+|T\d+)\]", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
_MIN_SENT_CHARS = 4


def split_sentences(answer: str) -> list[str]:
    """按中文句末标点/换行切句，过滤过短片段与纯符号。"""
    sentences: list[str] = []
    for chunk in _SENT_SPLIT.split(answer or ""):
        text = chunk.strip()
        text = re.sub(r"^[#>*\-\s]+", "", text).strip()
        if len(text) < _MIN_SENT_CHARS:
            continue
        if not re.search(r"[\w\u4e00-\u9fff]", text):
            continue
        sentences.append(text)
    return sentences


def extract_citations(sentence: str) -> list[str]:
    ids: list[str] = []
    for match in _CITE_RE.finditer(sentence or ""):
        eid = match.group(1).upper()
        if eid not in ids:
            ids.append(eid)
    return ids


@dataclass
class VerifyOutcome:
    confidence: float = 0.0
    raw_confidence: float = 0.0  # 覆盖率钳位前的置信度，供联合标定使用
    citation_coverage: float = 0.0
    claim_support: float = 0.0
    citations: list[dict] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    degraded: bool = False
    error: str = ""
    llm_trace: dict | None = None

    def to_dict(self) -> dict:
        return {
            "confidence": round(self.confidence, 4),
            "raw_confidence": round(self.raw_confidence, 4),
            "citation_coverage": round(self.citation_coverage, 4),
            "claim_support": round(self.claim_support, 4),
            "citations": len(self.citations),
            "invalid_citations": list(self.invalid_citations),
            "unsupported_claims": list(self.unsupported_claims)[:10],
            "degraded": self.degraded,
            "error": self.error,
        }


class Verifier:
    def __init__(
        self,
        brain=None,
        *,
        min_coverage: float = 0.60,
        verify_claims: bool = True,
        tau: float = 0.55,
    ):
        self.brain = brain
        self.min_coverage = max(0.0, min(1.0, float(min_coverage)))
        self.verify_claims = bool(verify_claims)
        self.tau = float(tau)

    # ---------- 对外 ----------
    def verify(
        self,
        *,
        answer: str,
        contexts: list[dict],
        tool_results: list[dict] | None = None,
        sufficiency: Sufficiency | None = None,
        conflicts: list[dict] | None = None,
        direct: bool = False,
        allow_llm: bool = True,
    ) -> VerifyOutcome:
        if direct:
            return VerifyOutcome(
                confidence=0.9,
                raw_confidence=0.9,
                citation_coverage=1.0,
                claim_support=1.0,
            )

        evidence_text: dict[str, str] = {
            str(c.get("evidence_id")): str(c.get("text", ""))
            for c in contexts
            if c.get("evidence_id")
        }
        for i, result in enumerate(ok_tool_results(tool_results), 1):
            evidence_text[f"T{i}"] = str(result.get("output", ""))
        valid_ids = set(evidence_text)

        sentences = split_sentences(answer)
        citations: list[dict] = []
        invalid: list[str] = []
        uncovered: list[str] = []
        claim_items: list[dict] = []
        covered = 0

        for sentence in sentences:
            cited = extract_citations(sentence)
            valid = [cid for cid in cited if cid in valid_ids]
            for cid in cited:
                if cid not in valid_ids and cid not in invalid:
                    invalid.append(cid)
            if valid:
                covered += 1
                claim_items.append(
                    {
                        "sentence": sentence,
                        "evidence_ids": valid,
                        "evidence": [
                            f"[{cid}] {evidence_text.get(cid, '')[:300]}"
                            for cid in valid
                        ],
                    }
                )
            else:
                uncovered.append(sentence)
            citations.append(
                {
                    "sentence": sentence,
                    "evidence_ids": cited,
                    "valid": bool(cited) and all(cid in valid_ids for cid in cited),
                }
            )

        total = len(sentences)
        coverage = covered / total if total else 0.0
        conflicts = conflicts or []

        claim_support = coverage
        unsupported = list(uncovered)
        degraded = False
        error = ""
        trace: dict | None = None

        if self.verify_claims and claim_items and self.brain is not None and allow_llm:
            messages = self._claim_messages(claim_items)
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
                if not data or not isinstance(data.get("claims"), list):
                    degraded = True
                    error = "json_parse_error"
                else:
                    supported = 0
                    seen = 0
                    for item in data.get("claims", [])[: len(claim_items)]:
                        if not isinstance(item, dict):
                            continue
                        seen += 1
                        if bool(item.get("supported")):
                            supported += 1
                        else:
                            idx = item.get("index")
                            if isinstance(idx, int) and 1 <= idx <= len(claim_items):
                                unsupported.append(claim_items[idx - 1]["sentence"])
                    if seen:
                        claim_support = supported / seen
                    else:
                        degraded = True
                        error = "empty_claims"
            except Exception as exc:
                degraded = True
                error = f"llm_error: {exc}"
        elif self.verify_claims and claim_items:
            degraded = True
            error = "no_brain"

        sufficiency_score = float(getattr(sufficiency, "score", 0.0) or 0.0)
        confidence = (
            0.40 * max(0.0, min(1.0, sufficiency_score))
            + 0.30 * max(0.0, min(1.0, claim_support))
            + 0.20 * max(0.0, min(1.0, coverage))
            + 0.10 * (0.0 if conflicts else 1.0)
        )
        raw_confidence = max(0.0, min(1.0, confidence))
        if coverage < self.min_coverage:
            confidence = min(confidence, self.tau - 0.01)
        confidence = max(0.0, min(1.0, confidence))

        return VerifyOutcome(
            confidence=confidence,
            raw_confidence=raw_confidence,
            citation_coverage=coverage,
            claim_support=claim_support,
            citations=citations,
            invalid_citations=invalid,
            unsupported_claims=list(dict.fromkeys(unsupported))[:20],
            degraded=degraded,
            error=error,
            llm_trace=trace,
        )

    # ---------- 内部 ----------
    @staticmethod
    def _claim_messages(claim_items: list[dict]) -> list[dict]:
        lines: list[str] = []
        for i, item in enumerate(claim_items, 1):
            evidence = "\n".join(item["evidence"]) or "（无）"
            lines.append(f"【句{i}】{item['sentence']}\n引用证据：\n{evidence}")
        user = "\n\n".join(lines)
        return [
            {"role": "system", "content": VERIFY_PROMPT},
            {"role": "user", "content": user},
        ]


def load_calibration(path: str | Path | None) -> dict:
    """读取阈值标定结果；文件缺失/损坏时返回空字典，使用配置默认值。"""
    if not path:
        return {}
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, float] = {}
    for key in ("min_coverage", "tau_answer_confidence"):
        if key in data:
            try:
                out[key] = float(data[key])
            except (TypeError, ValueError):
                continue
    return out


def build_verifier(cfg: dict, brain=None) -> Verifier:
    acfg = cfg.get("agent", {}) or {}
    ccfg = acfg.get("citations", {}) or {}
    calibration = load_calibration(ccfg.get("calibration_path"))
    return Verifier(
        brain,
        min_coverage=float(
            calibration.get("min_coverage", ccfg.get("min_coverage", 0.60))
        ),
        verify_claims=bool(ccfg.get("verify_claims", True)),
        tau=float(
            calibration.get(
                "tau_answer_confidence",
                ccfg.get("tau_answer_confidence", 0.55),
            )
        ),
    )


__all__ = [
    "VERIFY_PROMPT",
    "Verifier",
    "VerifyOutcome",
    "build_verifier",
    "extract_citations",
    "load_calibration",
    "split_sentences",
]
