"""B6 评测指标：EM/F1、拒答、引用/幻觉、检索、RAGAS 适配。

所有函数均为纯函数，便于单测与离线复现；每个指标在文档与消融表中都标注
评估器来源（programmatic / verifier / LLM-as-Judge / RAGAS）。
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

_PUNCT = set(
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "，。！？；：、“”‘’（）【】《》〈〉…—·～￥"
)
_CITATION_RE = re.compile(r"\[(?:E|T)\d+\]", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


# ==================== 文本与答案 ====================
def normalize_answer(text: Any, *, strip_citations: bool = True) -> str:
    """NFKC + 小写 + 去标点/空白；默认去掉 [E1]/[T1] 引用标记。"""
    s = unicodedata.normalize("NFKC", str(text or "")).lower()
    if strip_citations:
        s = _CITATION_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return "".join(ch for ch in s if ch not in _PUNCT).strip()


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_answer(text))


def exact_match(prediction: Any, gold: Any) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def f1_score(prediction: Any, gold: Any) -> float:
    """SQuAD 风格 token F1（中文按字、英文按词）。"""
    pred_tokens = _tokens(str(prediction or ""))
    gold_tokens = _tokens(str(gold or ""))
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ==================== 拒答 ====================
def refusal_metrics(rows: list[dict]) -> dict:
    """rows: [{"pred_refused": bool, "gold_refused": bool}]。

    以“是否应当拒答”为正类，输出 precision / recall / f1 / accuracy。
    """
    tp = fp = fn = tn = 0
    for row in rows:
        pred = bool(row.get("pred_refused"))
        gold = bool(row.get("gold_refused"))
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        else:
            tn += 1

    def _div(a: int, b: int) -> float:
        return round(a / b, 4) if b else 0.0

    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": _div(tp + tn, len(rows)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": len(rows),
        "evaluator": "programmatic（拒答规则/校验器）",
    }


# ==================== 引用与幻觉 ====================
def citation_metrics(rows: list[dict]) -> dict:
    """rows 字段：citation_coverage、citations[{valid}]、claims、unsupported_claims。"""
    coverages: list[float] = []
    total_citations = 0
    invalid_citations = 0
    claims = 0
    unsupported = 0
    for row in rows:
        if row.get("citation_coverage") is not None:
            coverages.append(float(row["citation_coverage"]))
        citations = row.get("citations") or []
        if citations:
            total_citations += len(citations)
            invalid_citations += sum(1 for c in citations if c.get("valid") is False)
        else:
            invalid_citations += int(row.get("invalid_citations", 0))
        claims += int(row.get("claims", 0))
        unsupported += int(row.get("unsupported_claims", 0))

    coverage = round(sum(coverages) / len(coverages), 4) if coverages else 0.0
    invalid_rate = (
        round(invalid_citations / total_citations, 4) if total_citations else 0.0
    )
    # 分母取 max(claims, unsupported)：unsupported 也属于 claim，避免比率 > 1
    hallucination_rate = (
        round(unsupported / max(claims, unsupported), 4)
        if max(claims, unsupported)
        else invalid_rate
    )
    return {
        "citation_coverage": coverage,
        "invalid_citation_rate": invalid_rate,
        "hallucination_rate": hallucination_rate,
        "total_citations": total_citations,
        "n": len(rows),
        "evaluator": "verifier（引用正则 + 模型辅助 claim 校验）",
    }


# ==================== 检索 ====================
def is_gold_hit(hit: dict, gold: list[dict]) -> bool:
    """与 eval_retrieval.py 保持同一命中口径：source 子串 + must_contain + page。"""
    source = str(hit.get("source", ""))
    text = str(hit.get("text") or hit.get("snippet") or "")
    for g in gold or []:
        if g.get("page") is not None and hit.get("page") != g["page"]:
            continue
        if g.get("source") and str(g["source"]) not in source:
            continue
        if g.get("must_contain") and str(g["must_contain"]) not in text:
            continue
        return True
    return False


def retrieval_metrics(
    ranked_lists: list[list[dict]], gold_lists: list[list[dict]], k: int = 5
) -> dict:
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for hits, gold in zip(ranked_lists, gold_lists):
        flags = [is_gold_hit(h, gold) for h in hits[:k]]
        rank = next((i + 1 for i, ok in enumerate(flags) if ok), 0)
        recalls.append(1.0 if rank else 0.0)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        # 二值相关性、单条理想证据：取首个命中的折损增益，避免多 gold/重复命中使 nDCG > 1
        ndcgs.append(1.0 / math.log2(rank + 1) if rank else 0.0)
    n = len(recalls) or 1
    return {
        "recall_at_k": round(sum(recalls) / n, 4),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "ndcg_at_k": round(sum(ndcgs) / n, 4),
        "k": k,
        "n": len(recalls),
        "evaluator": "programmatic（RETRIEVAL_GOLD 命中规则）",
    }


# ==================== 汇总 ====================
def aggregate_generation(rows: list[dict]) -> dict:
    """生成侧汇总：EM/F1 + 拒答 + 引用/幻觉 + 延迟与 token。"""
    answerable = [r for r in rows if not r.get("gold_refused")]
    em = (
        round(
            sum(exact_match(r.get("prediction", ""), r.get("gold", "")) for r in answerable)
            / len(answerable),
            4,
        )
        if answerable
        else 0.0
    )
    f1 = (
        round(
            sum(f1_score(r.get("prediction", ""), r.get("gold", "")) for r in answerable)
            / len(answerable),
            4,
        )
        if answerable
        else 0.0
    )
    latencies = [float(r["latency_ms"]) for r in rows if r.get("latency_ms") is not None]
    return {
        "exact_match": em,
        "f1": f1,
        "answerable_n": len(answerable),
        "refusal": refusal_metrics(rows),
        "citations": citation_metrics(rows),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "prompt_tokens": sum(int(r.get("prompt_tokens", 0)) for r in rows),
        "completion_tokens": sum(int(r.get("completion_tokens", 0)) for r in rows),
        "n": len(rows),
        "evaluator": {
            "exact_match": "programmatic（normalized EM）",
            "f1": "programmatic（token F1）",
            "refusal": "programmatic",
            "citations": "verifier",
        },
    }


# ==================== RAGAS 适配层（可选依赖） ====================
def ragas_scores(
    samples: list[dict],
    *,
    llm: Any = None,
    embeddings: Any = None,
) -> dict:
    """samples: [{"question","answer","contexts":[...],"ground_truth"}]。

    RAGAS 未安装或接口不兼容时返回 ``available=False``，不影响主评估流程。
    """
    try:
        import ragas  # noqa: F401
    except Exception:
        return {
            "available": False,
            "reason": "ragas_not_installed",
            "n": len(samples),
            "evaluator": "RAGAS（未安装，跳过）",
        }

    try:
        from ragas import evaluate

        try:  # RAGAS >= 0.2：EvaluationDataset + SingleTurnSample
            from ragas import EvaluationDataset, SingleTurnSample

            dataset = EvaluationDataset(
                samples=[
                    SingleTurnSample(
                        user_input=s.get("question", ""),
                        response=s.get("answer", ""),
                        retrieved_contexts=list(s.get("contexts") or []),
                        reference=s.get("ground_truth", ""),
                    )
                    for s in samples
                ]
            )
        except ImportError:  # 旧版：datasets.Dataset
            from datasets import Dataset

            dataset = Dataset.from_list(
                [
                    {
                        "question": s.get("question", ""),
                        "answer": s.get("answer", ""),
                        "contexts": list(s.get("contexts") or []),
                        "ground_truth": s.get("ground_truth", ""),
                    }
                    for s in samples
                ]
            )

        from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

        metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
        result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
        raw = dict(result)
        scores = {
            k: float(v)
            for k, v in raw.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        return {
            "available": True,
            "scores": scores,
            "n": len(samples),
            "evaluator": "RAGAS（Faithfulness/AnswerRelevancy/ContextP/R）",
        }
    except Exception as exc:  # 版本差异或服务不可用
        return {
            "available": False,
            "reason": f"ragas_error: {exc}",
            "n": len(samples),
            "evaluator": "RAGAS（调用失败，已跳过）",
        }


# ==================== IO ====================
def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
