"""生成质量评估：LLM-as-Judge 多模型对比（忠实度 / 相关性 / 拒答准确率）。

用法:
    python evaluation/eval_generation.py --models cloud,ft --out evaluation/results_llm_judge.json

实验设置：
- 回答模型：--models 指定的模型档案（cloud / ft ...），逐模型独立作答
- 评判模型：固定用主配置模型（云端强模型），保证裁判口径一致（论文需在实验设置中说明）
- 指标：
  * 忠实度 faithfulness (1-5)：答案是否忠于参考资料、无编造（仅统计知识内题）
  * 相关性 relevance (1-5)：是否切题回答（仅统计知识内题）
  * 拒答准确率：refusal_expected 题中被正确拒答的比例（答了不该答的 = 幻觉）
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402
from core.llm import LLMClient  # noqa: E402
from core.pipeline import build_pipeline  # noqa: E402

JUDGE_PROMPT = """你是严格的问答质量评审。请根据【参考资料】评审以下问答对：

【问题】{question}

【参考资料】
{contexts}

【答案】
{answer}

请仅输出 JSON（不要输出其他文字）：
{{"faithfulness": 1到5的整数(答案是否忠于参考资料、无编造), "relevance": 1到5的整数(是否切题回答了问题), "reason": "简要理由"}}"""

# SYSTEM_PROMPT 的拒答话术为"根据现有资料无法回答"。
# 注意不能只匹配"根据现有资料"——正常引用式回答（"根据现有资料，HNSW 是…"）会误报。
REFUSAL_PAT = ("无法回答", "资料中未提及", "资料中没有", "不足以回答", "无法从资料")


def build_clients(cfg: dict) -> dict[str, LLMClient]:
    """按主配置 + profiles 构建全部模型客户端。"""
    llm_cfg = cfg.get("llm", {})
    clients = {"cloud": LLMClient(llm_cfg)}
    for pid, prof in (llm_cfg.get("profiles") or {}).items():
        clients[pid] = LLMClient({**llm_cfg, **prof})
    return clients


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


def judge_one(judge: LLMClient, q: str, contexts: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=q, contexts=contexts or "（无）", answer=answer)
    try:
        return parse_json(judge.chat([{"role": "user", "content": prompt}]))
    except Exception as e:  # 网络失败等不中断整体评估
        return {"faithfulness": 0, "relevance": 0, "reason": f"judge 失败: {e}"}


def run_model(p, judge: LLMClient, client: LLMClient, model_id: str, items: list) -> dict:
    rows, refusals_total, refusals_ok = [], 0, 0
    f_sum = r_sum = t_sum = 0.0
    f_n = r_n = 0
    for item in items:
        q = item["question"]
        expected = bool(item.get("refusal_expected"))
        t0 = time.perf_counter()
        out = p.ask(q, llm=client)
        dt = time.perf_counter() - t0
        answer = out["answer"]
        contexts = "\n".join(f"[{i}] {r['snippet']}" for i, r in enumerate(out["references"], 1))
        verdict = judge_one(judge, q, contexts, answer)
        refused = any(pat in answer for pat in REFUSAL_PAT)
        rows.append(
            {
                "question": q,
                "refusal_expected": expected,
                "refused": refused,
                "faithfulness": verdict.get("faithfulness"),
                "relevance": verdict.get("relevance"),
                "reason": verdict.get("reason", ""),
                "answer": answer[:200],
                "latency_s": round(dt, 1),
            }
        )
        if expected:
            refusals_total += 1
            refusals_ok += int(refused)
        else:
            f_sum += verdict.get("faithfulness", 0)
            r_sum += verdict.get("relevance", 0)
            f_n += 1
            r_n += 1
        t_sum += dt
        mark = "（期望拒答）" if expected else ""
        print(
            f"  [{model_id}] {q}{mark} → 忠实度={verdict.get('faithfulness')} "
            f"相关性={verdict.get('relevance')} 拒答={refused} {dt:.1f}s"
        )
    return {
        "model": model_id,
        "faithfulness": round(f_sum / f_n, 2) if f_n else None,
        "relevance": round(r_sum / r_n, 2) if r_n else None,
        "refusal_correct": f"{refusals_ok}/{refusals_total}",
        "avg_latency_s": round(t_sum / len(items), 1) if items else 0,
        "details": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="evaluation/eval_set.json")
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", default="cloud,ft", help="逗号分隔的模型档案 id")
    ap.add_argument("--out", default=None, help="结果保存为 JSON（供论文引用）")
    args = ap.parse_args()

    items = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    cfg = load_config(args.config)
    p = build_pipeline(cfg)
    clients = build_clients(cfg)
    judge = clients["cloud"]  # 裁判固定用主配置（云端强模型）
    print(
        f"评估集: {args.eval_set}（{len(items)} 题，其中拒答题 "
        f"{sum(1 for i in items if i.get('refusal_expected'))}）"
        f" · 裁判模型 = {cfg['llm']['model']}"
    )

    results = []
    for mid in [m.strip() for m in args.models.split(",")]:
        if mid not in clients:
            print(f"  跳过未知模型档案: {mid}")
            continue
        print(f"\n—— 模型档案 [{mid}] 作答中 ——")
        results.append(run_model(p, judge, clients[mid], mid, items))

    print("\n===== 汇总（知识内题打分 / 拒答题计正确率）=====")
    for r in results:
        print(
            f"  {r['model']:<6} 忠实度 {r['faithfulness']}/5 · 相关性 {r['relevance']}/5"
            f" · 拒答正确 {r['refusal_correct']} · 平均 {r['avg_latency_s']}s/题"
        )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "judge_model": cfg["llm"]["model"],
                    "num_questions": len(items),
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
