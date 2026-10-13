"""命令行问答。

用法:
    python scripts/chat_cli.py "Milvus 是什么？"
    python scripts/chat_cli.py            # 进入交互模式
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402
from core.pipeline import build_pipeline  # noqa: E402


def ask_once(pipeline, question: str) -> None:
    out = pipeline.ask(question)
    print("\n答复：")
    print(out["answer"])
    print("\n参考来源：")
    for i, r in enumerate(out["references"], 1):
        print(f"  [{i}] {r['source']} 第{r['page']}页 (score={r['score']}, {'/'.join(r['channels'])})")
    print(f"\n耗时: {out['total_ms']}ms {out['timings']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", help="单次提问；省略则进入交互模式")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    pipeline = build_pipeline(load_config(args.config))
    if args.question:
        ask_once(pipeline, args.question)
        return
    print("交互问答模式（输入 q 退出）")
    while True:
        try:
            q = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"q", "quit", "exit"}:
            break
        ask_once(pipeline, q)


if __name__ == "__main__":
    main()
