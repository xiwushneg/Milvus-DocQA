"""E0–E4 Agentic RAG 消融实验。

变体（增量叠加，便于定位每项能力贡献）：
- E0 固定流水线（dense+BM25+RRF+rerank），agent 关闭
- E1 + 查询规划 / 意图路由 / 块类型路由
- E2 + 自反思闭环（充分性评估 → 缺口改写 → 再检索）
- E3 + 工具层（表格 DuckDB / 计算器 / 单位换算）
- E4 + 逐句引用校验 / 冲突并列 / 低置信拒答

检索基线只读取 evaluation/results_rerank_ablation.json，
禁止在脚本或文档中手工写入 0.902 / 0.917 等其它口径数字。
每个指标都在输出中标注评估器来源。

用法:
    python evaluation/ablation_agentic.py --eval-set evaluation/eval_set.json \
        --variants E0,E1,E2,E3,E4 --llm ft --out evaluation/results_agentic_ablation.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import (  # noqa: E402
    aggregate_generation,
    retrieval_metrics,
    write_json,
)

_STEPS: list[tuple[str, dict]] = [
    # E0 必须与 results_rerank_ablation.json 的 hybrid+rerank 基线同口径
    ("E0", {"agent": {"enabled": False}, "retrieval": {"rerank": {"enabled": True}}}),
    (
        "E1",
        {
            "agent": {
                "enabled": True,
                "max_rounds": 1,
                "hyde": {"enabled": False},
                "tools": {"enabled": False},
                "citations": {"enabled": False},
            }
        },
    ),
    ("E2", {"agent": {"max_rounds": 3}}),
    ("E3", {"agent": {"tools": {"enabled": True}}}),
    ("E4", {"agent": {"citations": {"enabled": True}}}),
]

LABELS = {
    "E0": "固定流水线（dense+BM25+RRF+rerank）",
    "E1": "+ 查询规划/意图路由/块类型",
    "E2": "+ 自反思闭环",
    "E3": "+ 工具层",
    "E4": "+ 逐句引用/冲突并列/低置信拒答",
}

EVALUATOR_LEGEND = {
    "retrieval_recall_at_k": "programmatic（eval_retrieval 命中规则，与 results_rerank_ablation.json 同口径）",
    "retrieval_mrr": "programmatic（同左）",
    "retrieval_ndcg_at_k": "programmatic（同左）",
    "exact_match": "programmatic（normalized EM）",
    "f1": "programmatic（token F1）",
    "refusal_precision/recall/f1": "programmatic（拒答规则）",
    "citation_coverage": "verifier（模型辅助 claim 校验）",
    "invalid_citation_rate": "verifier（引用正则）",
    "hallucination_rate": "verifier/LLM-as-Judge（unsupported claims）",
    "avg_latency_ms/tokens": "运行时计量（trace usage）",
    "ragas_*": "RAGAS（可选依赖，未安装则跳过）",
}

_REFUSAL_HINTS = ("无法可靠回答", "无法回答", "资料不足", "没有找到")


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def variant_config(base_cfg: dict, variant: str) -> dict:
    """按 E0→E4 顺序累积叠加配置覆盖。"""
    cfg = copy.deepcopy(base_cfg)
    for name, override in _STEPS:
        cfg = _deep_merge(cfg, override)
        if name == variant:
            return cfg
    raise ValueError(f"未知变体: {variant}")


def _gold_answer(item: dict) -> str:
    if item.get("answer"):
        return str(item["answer"])
    return "；".join(
        str(g.get("must_contain", "")) for g in item.get("gold") or [] if g.get("must_contain")
    )


def _pred_refused(out: dict, *, agent: bool) -> bool:
    """判定预测是否拒答：空答案一律算拒答。

    E0 无 ``refused`` 字段，按文本提示词匹配；E1+ 以 runtime 的
    ``refused`` 为准。两种口径下空/纯空白答案都视为未给出实质回答，
    避免空输出被统计成“未拒答”而虚高拒答召回。
    """
    answer = str(out.get("answer") or "").strip()
    if not answer:
        return True
    if agent:
        return bool(out.get("refused"))
    return any(hint in answer for hint in _REFUSAL_HINTS)


def _make_row(item: dict, out: dict, *, agent: bool) -> dict:
    metrics = out.get("metrics") or {}
    citations = out.get("citations") or []
    verify = next(
        (
            ev.get("payload", {})
            for ev in (out.get("trace") or [])
            if ev.get("stage") == "verify"
        ),
        {},
    )
    return {
        "question": item.get("question", ""),
        "prediction": out.get("answer", ""),
        "gold": _gold_answer(item),
        "gold_refused": bool(item.get("refusal_expected")),
        "pred_refused": _pred_refused(out, agent=agent),
        "references": out.get("references") or [],
        "citation_coverage": metrics.get("citation_coverage", verify.get("citation_coverage")),
        "citations": citations,
        "invalid_citations": metrics.get("invalid_citations", 0),
        "claims": verify.get("citations", len(citations)),
        "unsupported_claims": len(verify.get("unsupported_claims") or []),
        "latency_ms": metrics.get("latency_ms", out.get("total_ms")),
        "prompt_tokens": metrics.get("prompt_tokens", 0),
        "completion_tokens": metrics.get("completion_tokens", 0),
        "confidence": out.get("confidence", 0.0),
        "stop_reason": metrics.get("stop_reason", ""),
    }


def summarize(rows: list[dict], items: list[dict], top_k: int = 5) -> dict:
    """汇总生成/拒答/引用/检索指标（每个子指标均带 evaluator）。"""
    summary = aggregate_generation(rows)
    # 拒答题没有检索 gold，不计入 Recall/MRR/nDCG，避免拉低检索指标
    answerable_idx = [i for i, item in enumerate(items) if item.get("gold")]
    summary["retrieval"] = retrieval_metrics(
        [rows[i]["references"] for i in answerable_idx],
        [items[i]["gold"] for i in answerable_idx],
        k=top_k,
    )
    summary["avg_confidence"] = (
        round(sum(float(r.get("confidence", 0.0)) for r in rows) / len(rows), 4)
        if rows
        else 0.0
    )
    summary["stop_reasons"] = {}
    for row in rows:
        reason = row.get("stop_reason") or "-"
        summary["stop_reasons"][reason] = summary["stop_reasons"].get(reason, 0) + 1
    return summary


def load_retrieval_baseline(
    path: str | Path = "evaluation/results_rerank_ablation.json",
) -> dict:
    p = Path(path)
    if not p.is_file():
        return {"source": str(p), "available": False, "results": []}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {
        "source": str(p),
        "available": True,
        "evaluator": "evaluation/eval_retrieval.py（Recall@K / MRR）",
        "top_k": data.get("top_k"),
        "num_questions": data.get("num_questions"),
        "results": data.get("results", []),
    }


BASELINE_STRATEGY = "hybrid+rerank"


def compare_e0_to_baseline(
    variants: list[dict], baseline: dict, *, tol: float = 1e-3
) -> dict:
    """校验 E0 检索指标是否复现 hybrid+rerank 基线。

    同一知识库、同一 rerank 配置、同一评估集下，Recall@k / MRR 应在容差内一致；
    不一致说明语料、配置或命中口径发生了漂移。
    """
    e0 = next((v for v in variants if v.get("variant") == "E0"), None)
    if e0 is None:
        return {"performed": False, "passed": None, "reason": "E0 未运行"}

    expected_row = next(
        (
            r
            for r in baseline.get("results") or []
            if r.get("strategy") == BASELINE_STRATEGY
        ),
        None,
    )
    if not baseline.get("available") or expected_row is None:
        return {
            "performed": False,
            "passed": None,
            "reason": f"基线文件缺失或不含 {BASELINE_STRATEGY} 行",
        }

    actual = {
        "recall_at_k": float(e0["metrics"]["retrieval"]["recall_at_k"]),
        "mrr": float(e0["metrics"]["retrieval"]["mrr"]),
    }
    expected = {
        "recall_at_k": float(expected_row["recall_at_k"]),
        "mrr": float(expected_row["mrr"]),
    }
    passed = all(abs(actual[k] - expected[k]) <= tol for k in expected)
    return {
        "performed": True,
        "passed": passed,
        "expected": expected,
        "actual": actual,
        "tolerance": tol,
        "source": baseline.get("source"),
    }


def _run_e0(pipeline, question: str, client, top_k: int | None = None) -> dict:
    out = pipeline.ask(question, top_k=top_k, llm=client)
    return {
        "answer": out.get("answer", ""),
        "references": out.get("references", []),
        "metrics": {"latency_ms": out.get("total_ms", 0)},
        "trace": [],
        "refused": _pred_refused(out, agent=False),
    }


def run_ablation(
    items: list[dict],
    base_cfg: dict,
    *,
    variants: list[str] | None = None,
    runner_factory=None,
    top_k: int | None = None,
    baseline_path: str | Path = "evaluation/results_rerank_ablation.json",
    verify_baseline: bool = False,
) -> dict:
    """执行消融。

    ``runner_factory(name, cfg) -> Callable[[question], out]`` 可注入假执行器，
    单元测试无需真实 Milvus/LLM。``verify_baseline=True`` 时 E0 检索指标
    必须复现 hybrid+rerank 基线，否则抛错中止。
    """
    names = variants or [name for name, _ in _STEPS]
    k = int(top_k or (base_cfg.get("retrieval") or {}).get("final_top_k", 5))
    baseline = load_retrieval_baseline(baseline_path)
    results = []

    for name in names:
        cfg = variant_config(base_cfg, name)
        runner = runner_factory(name, cfg) if runner_factory else None
        if runner is None:
            raise ValueError("run_ablation 需要 runner_factory（CLI 已内置真实执行器）")
        rows = [
            _make_row(item, runner(item.get("question", "")), agent=name != "E0")
            for item in items
        ]
        results.append(
            {
                "variant": name,
                "label": LABELS[name],
                "metrics": summarize(rows, items, top_k=k),
            }
        )

    check = compare_e0_to_baseline(results, baseline)
    if verify_baseline:
        e0_ran = any(row["variant"] == "E0" for row in results)
        if not check.get("performed"):
            if e0_ran:
                raise RuntimeError(f"无法校验 E0 基线：{check.get('reason')}")
            # 子集运行（未包含 E0）允许继续，但显式标记基线校验已跳过；
            # 论文数字仍要求 E0 在场并完成校验。
            check["skipped"] = True
        elif not check["passed"]:
            raise RuntimeError(
                "E0 检索指标未复现 hybrid+rerank 基线："
                f"expected {check['expected']}, actual {check['actual']}。"
                "请确认知识库与 results_rerank_ablation.json 同源且 rerank 已启用；"
                "仅做探索时可加 --no-verify-baseline。"
            )

    return {
        "eval_set_n": len(items),
        "top_k": k,
        "retrieval_baseline": baseline,
        "baseline_check": check,
        "evaluator_legend": EVALUATOR_LEGEND,
        "variants": results,
    }


def _build_real_runners(base_cfg: dict, llm_id: str, top_k: int | None = None):
    from core.llm import LLMClient
    from core.pipeline import build_pipeline

    cfg = copy.deepcopy(base_cfg)
    # E0–E4 共用同一条检索链路；对照基线固定为 hybrid+rerank，
    # 因此无论 config.yaml 是否关闭重排，消融运行都显式启用。
    cfg.setdefault("retrieval", {}).setdefault("rerank", {})["enabled"] = True
    pipeline = build_pipeline(cfg)
    # 引用里的 snippet 截断到 200 字，用于展示；评测命中需按完整块文本判断
    lookup = {
        (r["source"], int(r["chunk_idx"])): r["text"]
        for r in pipeline.store.all_chunks()
    }

    def _enrich(out: dict) -> dict:
        for ref in out.get("references") or []:
            try:
                key = (ref.get("source"), int(ref.get("chunk_idx", -1)))
            except (TypeError, ValueError):
                continue
            text = lookup.get(key)
            if text:
                ref["text"] = text
        return out

    llms = {"cloud": {"client": LLMClient(cfg["llm"])}}
    profile = (cfg["llm"].get("profiles") or {}).get("ft")
    if profile:
        llms["ft"] = {"client": LLMClient({**cfg["llm"], **profile})}

    def factory(name: str, variant_cfg: dict):
        if name == "E0":
            runner = lambda q: _run_e0(  # noqa: E731
                pipeline, q, llms.get(llm_id, llms["cloud"])["client"], top_k=top_k
            )
        else:
            from core.agentic.runtime import build_agent_runtime

            runtime = build_agent_runtime(variant_cfg, pipeline, llms)
            runner = lambda q: runtime.ask(  # noqa: E731
                q, llm=llm_id, top_k=top_k, include_trace=True
            )
        return lambda q: _enrich(runner(q))

    return factory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="evaluation/eval_set.json")
    ap.add_argument("--variants", default="E0,E1,E2,E3,E4")
    ap.add_argument("--llm", default="cloud", help="回答模型档案：cloud | ft（ft 需本机推理服务）")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--out", default="evaluation/results_agentic_ablation.json")
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--verify-baseline",
        dest="verify_baseline",
        action="store_true",
        default=True,
        help="E0 检索指标必须复现 hybrid+rerank 基线，否则中止（默认开启）",
    )
    ap.add_argument(
        "--no-verify-baseline",
        dest="verify_baseline",
        action="store_false",
        help="跳过 E0 基线校验，仅用于探索性运行",
    )
    args = ap.parse_args()

    from app.config import load_config

    cfg = load_config(args.config)
    items = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    factory = _build_real_runners(cfg, args.llm, args.top_k)
    result = run_ablation(
        items,
        cfg,
        variants=[v.strip() for v in args.variants.split(",") if v.strip()],
        runner_factory=factory,
        top_k=args.top_k,
        verify_baseline=args.verify_baseline,
    )
    result["answer_llm"] = args.llm
    result["brain_llm"] = (cfg.get("agent") or {}).get("brain_llm")
    write_json(args.out, result)

    print(f"{'变体':<4} {'Recall@k':>9} {'MRR':>7} {'EM':>7} {'F1':>7} {'拒答F1':>7} {'覆盖':>6}")
    for row in result["variants"]:
        m = row["metrics"]
        print(
            f"{row['variant']:<4} {m['retrieval']['recall_at_k']:>9.3f}"
            f" {m['retrieval']['mrr']:>7.3f} {m['exact_match']:>7.3f}"
            f" {m['f1']:>7.3f} {m['refusal']['f1']:>7.3f}"
            f" {m['citations']['citation_coverage']:>6.3f}"
        )

    check = result["baseline_check"]
    if check.get("performed"):
        status = "通过" if check["passed"] else "未通过（已跳过中止）"
        print(f"基线校验：{status}（E0 vs {BASELINE_STRATEGY}）")
    else:
        print(f"基线校验：跳过（{check.get('reason')}）")
    print(f"\n结果已保存: {args.out}（检索基线来源：{result['retrieval_baseline']['source']}）")


if __name__ == "__main__":
    main()
