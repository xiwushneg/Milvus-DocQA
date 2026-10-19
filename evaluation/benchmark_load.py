"""并发压测：不同并发级别下的成功率 / 延迟分位数 / 吞吐。

用法:
    python evaluation/benchmark_load.py --concurrency 1,5,10 --requests 30 --mode total,ttft --llm ft

模式:
    total  非流式 /api/chat 端到端延迟
    ttft   流式 /api/chat/stream 首字延迟（Time-To-First-Token，用户体感指标）

说明: 问题取自评估集前若干题轮转，保证检索真实触发。
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")


def pct(values: list, p: int) -> float:
    s = sorted(values)
    return s[min(int(len(s) * p / 100), len(s) - 1)] if s else 0.0


def one_request(api: str, question: str, llm: str, mode: str, timeout: int) -> dict:
    t0 = time.perf_counter()
    if mode == "ttft":
        first = None
        with requests.post(
            f"{api}/api/chat/stream",
            json={"question": question, "llm": llm, "history": []},
            stream=True, timeout=timeout,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    ev = json.loads(line[5:].strip())
                    if ev.get("type") == "delta":
                        first = time.perf_counter() - t0
                        break
            for _ in r.iter_lines():  # 读完剩余事件，释放连接
                pass
        return {"ok": True, "ttft": first, "total": time.perf_counter() - t0}
    r = requests.post(
        f"{api}/api/chat",
        json={"question": question, "llm": llm},
        timeout=timeout,
    )
    r.raise_for_status()
    return {"ok": True, "total": time.perf_counter() - t0}


def run_level(api: str, questions: list, llm: str, mode: str, conc: int, n_req: int, timeout: int) -> dict:
    results = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futures = [
            ex.submit(one_request, api, questions[i % len(questions)], llm, mode, timeout)
            for i in range(n_req)
        ]
        for f in futures:
            try:
                results.append(f.result())
            except Exception:
                results.append({"ok": False})
    wall = time.perf_counter() - t0

    ok = [r for r in results if r.get("ok")]
    totals = [r["total"] for r in ok]
    row = {
        "mode": mode,
        "concurrency": conc,
        "requests": n_req,
        "success_rate": round(len(ok) / n_req, 3),
        "p50_s": round(pct(totals, 50), 2),
        "p95_s": round(pct(totals, 95), 2),
        "p99_s": round(pct(totals, 99), 2),
        "throughput_rps": round(n_req / wall, 2),
    }
    if mode == "ttft":
        ttfts = [r["ttft"] for r in ok if r.get("ttft") is not None]
        row["ttft_p50_s"] = round(pct(ttfts, 50), 2)
        row["ttft_p95_s"] = round(pct(ttfts, 95), 2)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--eval-set", default="evaluation/eval_set.json")
    ap.add_argument("--llm", default="ft", help="压测目标模型档案（默认 ft 本机服务）")
    ap.add_argument("--concurrency", default="1,5,10", help="逗号分隔的并发梯度")
    ap.add_argument("--requests", type=int, default=30, help="每级请求数")
    ap.add_argument("--mode", default="total,ttft")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default=None, help="结果保存为 JSON")
    args = ap.parse_args()

    questions = [
        item["question"]
        for item in json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
        if not item.get("refusal_expected")
    ][:10]
    levels = [int(x) for x in args.concurrency.split(",")]
    modes = [m.strip() for m in args.mode.split(",")]

    print(f"压测目标: {args.api}（llm={args.llm}）· 问题池 {len(questions)} 题 · 每级 {args.requests} 请求")
    all_rows = []
    for mode in modes:
        print(f"\n—— 模式 [{mode}] ——")
        header = (
            "  并发   成功率    P50      P95      P99     吞吐(req/s)"
            + ("   首字P50  首字P95" if mode == "ttft" else "")
        )
        print(header)
        for conc in levels:
            row = run_level(args.api, questions, args.llm, mode, conc, args.requests, args.timeout)
            all_rows.append(row)
            base = (
                f"  {row['concurrency']:<4}  {row['success_rate']:<6}  "
                f"{row['p50_s']:<7.2f}  {row['p95_s']:<7.2f}  {row['p99_s']:<7.2f}  {row['throughput_rps']:<10.2f}"
            )
            if mode == "ttft":
                base += f"  {row['ttft_p50_s']:<7.2f}  {row['ttft_p95_s']:<7.2f}"
            print(base)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"llm": args.llm, "requests_per_level": args.requests, "results": all_rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
