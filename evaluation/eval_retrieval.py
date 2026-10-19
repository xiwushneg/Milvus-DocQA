"""检索效果评估：Recall@K 与 MRR，对比 纯稠密 / 混合(RRF) / 混合+重排 三种策略。

用法:
    python evaluation/build_eval_set.py                 # 先生成评测集模板
    python evaluation/eval_retrieval.py --top-k 5       # 运行评估

指标定义：
- Recall@K：答案命中片段出现在 Top-K 中的问题占比
- MRR：首个命中片段排名倒数的均值
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402
from core.fusion import rrf_fuse  # noqa: E402
from core.pipeline import build_pipeline  # noqa: E402


def is_gold_hit(hit: dict, gold: list[dict]) -> bool:
    for g in gold:
        if g["source"] in hit.get("source", "") and g.get("must_contain", "") in hit.get("text", ""):
            return True
    return False


def evaluate_strategy(name: str, rows: list, top_k: int) -> dict:
    recalls, ranks, elapsed = [], [], []
    for hits, gold, ms in rows:
        hit_ranks = [i + 1 for i, h in enumerate(hits) if is_gold_hit(h, gold)]
        recalls.append(1.0 if hit_ranks else 0.0)
        ranks.append(1.0 / hit_ranks[0] if hit_ranks else 0.0)
        elapsed.append(ms)
    n = len(recalls) or 1
    result = {
        "strategy": name,
        "recall_at_k": round(sum(recalls) / n, 3),
        "mrr": round(sum(ranks) / n, 3),
        "avg_ms": round(sum(elapsed) / n, 1),
        "top_k": top_k,
    }
    print(
        f"  {name:<16} Recall@{top_k} = {result['recall_at_k']:.3f}   "
        f"MRR = {result['mrr']:.3f}   平均耗时 = {result['avg_ms']}ms/题"
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="evaluation/eval_set.json")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--rerank", action="store_true",
        help="启用重排序策略（覆盖 config 中 rerank.enabled=false）",
    )
    ap.add_argument("--out", default=None, help="结果保存为 JSON（供论文引用）")
    args = ap.parse_args()

    eval_set = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    cfg = load_config(args.config)
    p = build_pipeline(cfg)
    p._ensure_bm25()
    if args.rerank:
        p.reranker.enabled = True
        print(f"重排序: {p.reranker.model}（mode={p.reranker.mode}）")
    rrf_k = int(cfg["retrieval"].get("rrf_k", 60))
    fetch_k = max(args.top_k * 2, 10)

    strat_dense, strat_hybrid, strat_rerank = [], [], []
    for item in eval_set:
        if item.get("refusal_expected"):
            continue  # 拒答题无检索 gold，仅用于生成评估（eval_generation.py）
        q, gold = item["question"], item["gold"]
        t0 = time.perf_counter()
        qvec = p.embedder.embed_query(q)
        dense = p.store.search_dense(qvec, top_k=fetch_k)
        t1 = time.perf_counter()
        bm25 = p.bm25.search(q, top_k=fetch_k)
        hybrid = rrf_fuse([dense, bm25], k=rrf_k)
        t2 = time.perf_counter()
        strat_dense.append((dense[: args.top_k], gold, (t1 - t0) * 1000))
        strat_hybrid.append((hybrid[: args.top_k], gold, (t2 - t0) * 1000))
        if p.reranker.enabled:
            reranked = p.reranker.rerank(q, hybrid, top_n=args.top_k)
            t3 = time.perf_counter()
            # 重排策略端到端耗时 = 前置混合检索 + 重排调用
            strat_rerank.append((reranked, gold, (t3 - t0) * 1000))

    print(f"评估集: {args.eval_set}（{len(eval_set)} 题）· embedding = {p.embedder.name}")
    results = [
        evaluate_strategy("dense", strat_dense, args.top_k),
        evaluate_strategy("hybrid(RRF)", strat_hybrid, args.top_k),
    ]
    if strat_rerank:
        results.append(evaluate_strategy("hybrid+rerank", strat_rerank, args.top_k))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"top_k": args.top_k, "num_questions": len(eval_set), "results": results},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n结果已保存: {args.out}")
    print("\n提示：将三种策略结果填入论文实验对比表，并分析混合检索对专有名词类问题的增益。")


if __name__ == "__main__":
    main()
