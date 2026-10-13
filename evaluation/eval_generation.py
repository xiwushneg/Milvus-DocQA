"""生成质量评估：LLM-as-Judge 对答案的忠实度与相关性打分（1-5）。

用法:
    python evaluation/eval_generation.py --eval-set evaluation/eval_set.json

说明：评判模型默认与回答模型相同；论文中建议用更强的模型（如 Qwen2.5-72B）作
Judge 以提升评审判定的可信度，并在实验设置一节说明。
"""
import argparse
import json
import re
import sys
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


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="evaluation/eval_set.json")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    eval_set = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    cfg = load_config(args.config)
    p = build_pipeline(cfg)
    judge = LLMClient(cfg.get("llm", {}))

    scores: list[dict] = []
    for item in eval_set:
        q = item["question"]
        out = p.ask(q)
        contexts = "\n".join(f"[{i}] {r['snippet']}" for i, r in enumerate(out["references"], 1))
        prompt = JUDGE_PROMPT.format(
            question=q, contexts=contexts or "（无）", answer=out["answer"]
        )
        try:
            verdict = parse_json(judge.chat([{"role": "user", "content": prompt}]))
        except Exception as e:  # 网络失败等不中断整体评估
            verdict = {"faithfulness": 0, "relevance": 0, "reason": f"judge 失败: {e}"}
        row = {"question": q, **verdict}
        scores.append(row)
        print(
            f"  {q} → faithfulness={verdict.get('faithfulness')} "
            f"relevance={verdict.get('relevance')}"
        )

    if scores:
        f_mean = sum(s.get("faithfulness", 0) for s in scores) / len(scores)
        r_mean = sum(s.get("relevance", 0) for s in scores) / len(scores)
        print(f"\n平均忠实度 {f_mean:.2f}/5，平均相关性 {r_mean:.2f}/5（{len(scores)} 题）")


if __name__ == "__main__":
    main()
