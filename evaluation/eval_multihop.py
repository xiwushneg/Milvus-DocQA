"""2WikiMultiHopQA / HotpotQA 内存评测（无需 Milvus / 外部服务）。

输入 JSONL/JSON，每题字段兼容两种数据集：
- contexts: [{"title": "...", "text": "..."}]（2Wiki / Hotpot 均可整理成该格式）
- supporting_titles: ["..."] 或 supporting_facts: [["title", sent_id], ...]
- answer: 参考答案
- prediction / pred_contexts: 可选，系统输出；缺省时仅用内存 BM25 评估检索

用法:
    python evaluation/eval_multihop.py --input data/hotpot_dev.jsonl --top-k 5 \
        --out evaluation/results_multihop.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.bm25 import BM25Index  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    exact_match,
    f1_score,
    retrieval_metrics,
    write_json,
)


def load_items(path: str | Path) -> list[dict]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        return data if isinstance(data, list) else data.get("data", [])
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def gold_titles(item: dict) -> list[str]:
    titles = list(item.get("supporting_titles") or [])
    for fact in item.get("supporting_facts") or []:
        if isinstance(fact, (list, tuple)) and fact:
            titles.append(str(fact[0]))
        elif isinstance(fact, dict) and fact.get("title"):
            titles.append(str(fact["title"]))
    for ctx in item.get("contexts") or []:
        if ctx.get("is_supporting") or ctx.get("supporting"):
            titles.append(str(ctx.get("title", "")))
    # 去重保序
    out: list[str] = []
    for t in titles:
        if t and t not in out:
            out.append(t)
    return out


def rank_contexts(question: str, contexts: list[dict], top_k: int) -> list[dict]:
    """用内存 BM25 对候选段落排序；零命中时退回原始顺序。"""
    rows = [
        {
            "id": i,
            "text": f"{c.get('title', '')} {c.get('text', '')}",
            "source": c.get("title", ""),
            "page": 0,
            "chunk_idx": i,
        }
        for i, c in enumerate(contexts)
    ]
    index = BM25Index()
    index.build(rows)
    hits = index.search(question, top_k=top_k)
    if not hits:
        return contexts[:top_k]
    return hits


def evaluate(items: list[dict], top_k: int = 5) -> dict:
    ranked_lists: list[list[dict]] = []
    gold_lists: list[list[dict]] = []
    ems: list[float] = []
    f1s: list[float] = []
    has_predictions = False

    for item in items:
        contexts = item.get("pred_contexts")
        if not contexts:
            contexts = rank_contexts(
                item.get("question", ""), item.get("contexts") or [], top_k
            )
        contexts = [
            {
                "source": c.get("source") or c.get("title", ""),
                "text": c.get("text", ""),
                "page": c.get("page", 0),
            }
            for c in contexts
        ]
        ranked_lists.append(contexts[:top_k])
        gold_lists.append(
            [{"source": t} for t in gold_titles(item)]
        )

        if item.get("prediction") is not None and item.get("answer") is not None:
            has_predictions = True
            ems.append(exact_match(item["prediction"], item["answer"]))
            f1s.append(f1_score(item["prediction"], item["answer"]))

    result = {
        "n": len(items),
        "top_k": top_k,
        "retrieval": retrieval_metrics(ranked_lists, gold_lists, k=top_k),
        "has_predictions": has_predictions,
        "answer": {
            "exact_match": round(sum(ems) / len(ems), 4) if ems else None,
            "f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
            "n": len(ems),
            "evaluator": "programmatic（normalized EM / token F1）",
        },
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="2Wiki/Hotpot 风格 JSON 或 JSONL")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items = load_items(args.input)
    result = evaluate(items, top_k=args.top_k)
    r = result["retrieval"]
    print(
        f"{args.input}: {result['n']} 题 · Recall@{args.top_k} = {r['recall_at_k']:.3f}"
        f" · MRR = {r['mrr']:.3f} · nDCG@{args.top_k} = {r['ndcg_at_k']:.3f}"
    )
    if result["has_predictions"]:
        a = result["answer"]
        print(f"答案：EM = {a['exact_match']:.3f} · F1 = {a['f1']:.3f}")
    else:
        print("未提供 prediction，仅评估检索；答案 EM/F1 跳过。")
    if args.out:
        write_json(args.out, result)
        print(f"结果已保存: {args.out}")


if __name__ == "__main__":
    main()
