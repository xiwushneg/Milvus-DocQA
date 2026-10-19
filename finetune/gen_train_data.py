"""微调训练数据生成器：文档 → 分块 → 大模型生成问答对 → LLaMA-Factory Alpaca 格式。

训练样本的 prompt 结构与运行时 RAG 生成完全一致（复用 core/llm.py 的
SYSTEM_PROMPT 与【参考资料】/【问题】格式），保证"训练分布 = 推理分布"。

用法（在项目虚拟环境中运行）:
    python finetune/gen_train_data.py --path scripts/demo_data
    python finetune/gen_train_data.py --path D:\\我的文档库 --per-chunk 3
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402
from core.llm import LLMClient, SYSTEM_PROMPT  # noqa: E402
from core.loader import parse_file  # noqa: E402
from core.splitter import split_segments  # noqa: E402

REFUSAL_ANSWER = "根据现有资料无法回答。"
OFFTOPIC_QUESTIONS = [
    "今天天气怎么样？",
    "推荐一部好看的电影。",
    "怎么提高英语口语水平？",
]

GEN_PROMPT = """你是数据标注专家。请根据下面的资料片段，生成 {n} 个中文问答对，用于训练文档问答助手。

要求：
1. 答案必须完全依据资料内容，不得引入资料外知识，不得编造；
2. 答案中的关键结论后标注引用 [1]（资料片段是唯一参考）；
3. 问题类型尽量多样：事实型、解释型、比较型、操作型；
4. 只输出 JSON 数组，不要输出其他任何内容，格式：
[{{"question": "问题", "answer": "答案"}}]

【资料片段】
{chunk}"""


def extract_json(text: str) -> list[dict] | None:
    """从模型输出中稳健地提取 JSON 数组（容忍 ```json 围栏）。"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    if m:
        text = m.group(1)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 LoRA 微调训练数据")
    ap.add_argument("--path", default="scripts/demo_data", help="文档文件或目录")
    ap.add_argument("--per-chunk", type=int, default=2, help="每个分块生成几条问答")
    ap.add_argument("--min-chars", type=int, default=60, help="过短的分块跳过")
    ap.add_argument("--refusal-ratio", type=float, default=0.15,
                    help="负样本（拒答）比例，教会模型'不知道就说不知道'")
    ap.add_argument("--out", default="finetune/data/docqa_lora.json")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    llm = LLMClient(cfg["llm"])
    print(f"教师模型 = {cfg['llm']['model']}  @ {cfg['llm']['base_url']}")

    p = Path(args.path)
    if p.is_dir():
        files = [f for f in sorted(p.rglob("*")) if f.suffix.lower() in
                 {".pdf", ".docx", ".md", ".markdown", ".txt", ".text"}]
    else:
        files = [p]
    if not files:
        print("未找到文档")
        return

    samples: list[dict] = []
    rng = random.Random(42)
    for f in files:
        try:
            segments = parse_file(str(f))
        except Exception as e:  # noqa: BLE001
            print(f"  跳过 {f.name}: {e}")
            continue
        ck = cfg["chunking"]
        chunks = split_segments(segments, f.name, ck["chunk_size"],
                                ck["chunk_overlap"], ck["min_chunk_size"])
        chunks = [c for c in chunks if len(c.text) >= args.min_chars]
        print(f"  {f.name}: {len(segments)} 段 → {len(chunks)} 块（有效）")
        for c in chunks:
            # --- 正样本：依据分块生成问答对 ---
            user = GEN_PROMPT.format(n=args.per_chunk, chunk=c.text)
            try:
                raw = llm.chat([{"role": "user", "content": user}])
            except Exception as e:  # noqa: BLE001
                print(f"    生成失败，跳过该块: {e}")
                continue
            qa_list = extract_json(raw) or []
            for qa in qa_list:
                q, a = str(qa.get("question", "")).strip(), str(qa.get("answer", "")).strip()
                if not q or not a:
                    continue
                samples.append({
                    "instruction": f"【参考资料】\n[1] （来源: {c.source} 第{c.page}页）\n{c.text}"
                                   f"\n\n【问题】\n{q}",
                    "system": SYSTEM_PROMPT,
                    "input": "",
                    "output": a,
                })
            time.sleep(0.3)  # 轻微限速，避免 API 限流

            # --- 负样本：资料外问题 → 教拒答 ---
            if rng.random() < args.refusal_ratio:
                q = rng.choice(OFFTOPIC_QUESTIONS)
                samples.append({
                    "instruction": f"【参考资料】\n[1] （来源: {c.source} 第{c.page}页）\n{c.text}"
                                   f"\n\n【问题】\n{q}",
                    "system": SYSTEM_PROMPT,
                    "input": "",
                    "output": REFUSAL_ANSWER,
                })

    rng.shuffle(samples)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成：共 {len(samples)} 条样本 → {out}")
    print("注意：建议 ≥300 条。当前文档较少时，请扩充 --path 指向的文档库后重新生成，"
          "并人工抽查 20% 样本质量（答案是否忠于资料、引用是否正确）。")


if __name__ == "__main__":
    main()
