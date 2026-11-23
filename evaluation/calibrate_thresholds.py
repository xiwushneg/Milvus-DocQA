"""拒答阈值标定：基于开发集 ``confidence / answerable`` 网格搜索 tau。

若数据同时包含 ``raw_confidence`` 与 ``coverage``（新版 export 脚本输出），
则对 ``(tau, min_coverage)`` 做联合搜索，复现校验器的覆盖率钳位：
``coverage < min_coverage`` 时置信度被压到 ``tau - 0.01``。

输入 JSONL 每行：``{"confidence": 0.62, "answerable": true}``
输出默认 ``data/calibration.json``，``AgentRuntime`` 启动时自动读取并覆盖
``citations.tau_answer_confidence`` / ``citations.min_coverage``。

用法:
    python evaluation/calibrate_thresholds.py --input data/dev_confidence.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _score(
    pairs: list[dict],
    tau: float,
    *,
    min_coverage: float | None = None,
) -> dict:
    """按候选阈值计算拒答指标；给定 min_coverage 时复现钳位逻辑。"""
    refused: list[dict] = []
    for p in pairs:
        raw = float(p.get("raw_confidence", p["confidence"]))
        if min_coverage is not None and float(p.get("coverage", 1.0)) < min_coverage:
            raw = min(raw, tau - 0.01)
        if raw < tau:
            refused.append(p)
    unanswerable = [p for p in pairs if not p.get("answerable")]
    answerable = [p for p in pairs if p.get("answerable")]
    tp = len([p for p in refused if not p.get("answerable")])
    fp = len(refused) - tp
    fn = len(unanswerable) - tp
    false_refusal = (
        len([p for p in refused if p.get("answerable")]) / len(answerable)
        if answerable
        else 0.0
    )
    record = {
        "tau_answer_confidence": tau,
        "refusal_f1": round(_f1(tp, fp, fn), 4),
        "false_refusal_rate": round(false_refusal, 4),
        "refused": len(refused),
    }
    if min_coverage is not None:
        record["min_coverage"] = round(min_coverage, 4)
    return record


def _coverage_candidates(pairs: list[dict]) -> list[float]:
    """由观测覆盖率生成钳位阈值候选（取相邻值中点，覆盖全部分段）。"""
    values = sorted({round(float(p.get("coverage", 0.0)), 4) for p in pairs})
    candidates = [0.0]
    for left, right in zip(values, values[1:]):
        candidates.append(round((left + right) / 2, 4))
    if values:
        candidates.append(values[-1])
        candidates.append(min(1.0, round(values[-1] + 0.01, 4)))
    return sorted(set(candidates))


def calibrate(
    pairs: list[dict],
    *,
    grid_step: float = 0.01,
    max_false_refusal: float = 0.05,
) -> dict:
    """挑选假拒答率 ≤ 上限前提下拒答 F1 最大的阈值组合。"""
    if not pairs:
        raise ValueError("标定数据为空")
    answerable = [p for p in pairs if p.get("answerable")]
    unanswerable = [p for p in pairs if not p.get("answerable")]

    joint = all(
        "raw_confidence" in p and "coverage" in p for p in pairs
    )
    min_cov_grid: list[float | None] = (
        _coverage_candidates(pairs) if joint else [None]
    )

    best: dict | None = None
    fallback: dict | None = None
    steps = int(round(1.0 / grid_step))
    for min_cov in min_cov_grid:
        for i in range(1, steps):
            tau = round(i * grid_step, 4)
            record = _score(pairs, tau, min_coverage=min_cov)
            if (
                fallback is None
                or record["false_refusal_rate"] < fallback["false_refusal_rate"]
            ):
                fallback = record
            if record["false_refusal_rate"] <= max_false_refusal and (
                best is None or record["refusal_f1"] > best["refusal_f1"]
            ):
                best = record

    chosen = best or fallback or {"tau_answer_confidence": 0.55}
    return {
        **chosen,
        "mode": "joint" if joint else "tau_only",
        "n": len(pairs),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "max_false_refusal": max_false_refusal,
    }


def _load_jsonl(path: str | Path) -> list[dict]:
    pairs: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        item = json.loads(text)
        pair = {
            "confidence": float(item["confidence"]),
            "answerable": bool(item["answerable"]),
        }
        if "raw_confidence" in item:
            pair["raw_confidence"] = float(item["raw_confidence"])
        if "coverage" in item:
            pair["coverage"] = float(item["coverage"])
        pairs.append(pair)
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="拒答阈值标定")
    ap.add_argument("--input", required=True, help="开发集 JSONL: confidence/answerable")
    ap.add_argument("--output", default="data/calibration.json")
    ap.add_argument("--grid-step", type=float, default=0.01)
    ap.add_argument("--max-false-refusal", type=float, default=0.05)
    args = ap.parse_args()

    result = calibrate(
        _load_jsonl(args.input),
        grid_step=args.grid_step,
        max_false_refusal=args.max_false_refusal,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"已写入 {target}")


if __name__ == "__main__":
    main()
