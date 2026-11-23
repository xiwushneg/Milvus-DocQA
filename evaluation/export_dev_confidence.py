"""导出 dev split 的 E4 置信度，供拒答阈值标定使用。

只使用开发集，排除测试集问题；输出 JSONL 每行:
``{"question": ..., "confidence": 0.73, "answerable": true, "refused": false}``

用法:
    python evaluation/export_dev_confidence.py \
        --config config/config_baseline116.yaml \
        --n 100 --n-refusal 10 \
        --out data/dev_confidence.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402
from evaluation.ablation_agentic import _build_real_runners, variant_config  # noqa: E402


def _verify_payload(out: dict) -> dict:
    """返回最后一个 verify 事件 payload；早拒场景为空 dict。"""
    payload: dict = {}
    for ev in out.get("trace") or []:
        if ev.get("stage") == "verify":
            payload = ev.get("payload") or {}
    return payload


def _raw_confidence(out: dict) -> float:
    """取校验阶段的原始置信度（优先覆盖率钳位前的 raw_confidence）。

    ``refuse`` 节点会把 ``state.confidence`` 置 0，直接读返回值会让低置信拒答
    全部变成 0，无法参与阈值标定；原始值保存在 trace 的 ``verify`` 事件里。
    早拒（out_of_scope / 无证据）没有 verify 事件，置信度按 0 处理。
    旧 trace 无 ``raw_confidence`` 时回退到 ``confidence``。
    """
    conf = float(out.get("confidence") or 0.0)
    payload = _verify_payload(out)
    if "raw_confidence" in payload:
        return float(payload.get("raw_confidence") or 0.0)
    if "confidence" in payload:
        return float(payload.get("confidence") or 0.0)
    return conf


def _load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def sample_dev(
    rows: list[dict],
    *,
    n: int,
    n_refusal: int,
    seed: int,
    exclude: set[str],
) -> list[dict]:
    answerable = [r for r in rows if not r.get("refusal_expected") and r["question"] not in exclude]
    refusal = [r for r in rows if r.get("refusal_expected") and r["question"] not in exclude]
    if len(refusal) < n_refusal:
        raise ValueError(f"拒答样本不足：需要 {n_refusal}，只有 {len(refusal)}")
    need_answerable = n - n_refusal
    if len(answerable) < need_answerable:
        raise ValueError(f"可答样本不足：需要 {need_answerable}，只有 {len(answerable)}")

    rng = random.Random(seed)
    picked = rng.sample(refusal, n_refusal) + rng.sample(answerable, need_answerable)
    rng.shuffle(picked)
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description="导出 dev 置信度")
    ap.add_argument("--dev-set", default="evaluation/qa_zh_300.jsonl")
    ap.add_argument("--test-set", default="evaluation/eval_set.json")
    ap.add_argument("--config", default="config/config_baseline116.yaml")
    ap.add_argument("--llm", default="cloud", help="回答模型档案：cloud | ft")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-refusal", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/dev_confidence.jsonl")
    args = ap.parse_args()

    rows = _load_jsonl(args.dev_set)
    test = json.loads(Path(args.test_set).read_text(encoding="utf-8"))
    dev = sample_dev(
        rows,
        n=args.n,
        n_refusal=args.n_refusal,
        seed=args.seed,
        exclude={item["question"] for item in test},
    )

    cfg = load_config(args.config)
    factory = _build_real_runners(cfg, args.llm)
    runner = factory("E4", variant_config(cfg, "E4"))

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for i, item in enumerate(dev, 1):
            try:
                out = runner(item["question"])
            except Exception as exc:  # 单题失败不阻塞标定，记录后跳过
                print(f"[{i}/{len(dev)}] 失败: {item['question']} -> {exc}")
                continue
            payload = _verify_payload(out)
            raw_conf = round(_raw_confidence(out), 4)
            record = {
                "question": item["question"],
                "confidence": raw_conf,
                "raw_confidence": raw_conf,
                "coverage": round(float(payload.get("citation_coverage", 0.0) or 0.0), 4),
                "answerable": not item.get("refusal_expected", False),
                "refused": bool(out.get("refused")),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{i}/{len(dev)}] conf={record['confidence']:.3f} "
                f"answerable={record['answerable']} refused={record['refused']}"
            )

    print(f"已写入 {target}（{len(dev)} 题）")


if __name__ == "__main__":
    main()
