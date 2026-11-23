"""构建中文 QA 评测集（默认 300 题）。

两种模式：
- template：离线确定性生成，按块抽取原文关键短语，零 API 依赖，用于规模回归与阈值标定；
- llm：调用 brain 模型从块生成自然问题，质量更高，需要可用的 LLM API。

输出 JSONL，字段与 evaluation/eval_set.json 一致：
    {"question": "...", "gold": [{"source": "...", "must_contain": "..."}],
     "refusal_expected": false}

注意：template 集的问题包含原文关键短语，检索偏易，仅用于回归/稳定性评估，
不能替代论文中的 20 题人工 gold 检索基线。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.splitter import recursive_split  # noqa: E402

QA_PROMPT = """你是中文 RAG 评测集标注员。请根据【文档片段】写 1 个可由该片段直接回答的问题，\
并给出答案中必须出现的原文短语。

要求：
1. 问题自然、完整，不得包含答案短语本身；
2. must_contain 必须逐字出现在片段中，长度 4-30 字；
3. 只输出 JSON，不要解释：{{"question": "...", "must_contain": "..."}}

【文档】{source}
【片段】
{text}"""

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
_MD_PREFIX = re.compile(r"^[#>*\-\s|]+")


def clean_doc_name(name: str) -> str:
    return re.sub(r"\.[a-zA-Z0-9]+$", "", name).strip()


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    return [p for p in parts if not p.startswith("|") and not p.startswith("---")]


def pick_keyphrases(text: str, limit: int = 4) -> list[str]:
    """从块中抽取互不相同的原文短语，优先含数字/英文/书名号的句子。"""
    candidates: list[str] = []
    for sent in _sentences(text):
        clean = _MD_PREFIX.sub("", sent).strip()
        if len(clean) < 10:
            continue
        score = (
            2 * bool(re.search(r"[A-Za-z0-9]", clean))
            + bool(re.search(r"[《》“”]", clean))
            + min(len(clean), 60) / 60
        )
        candidates.append((score, clean[:40]))
    candidates.sort(key=lambda x: x[0], reverse=True)

    out: list[str] = []
    for _, phrase in candidates:
        if phrase in out:
            continue
        # 避免短语互为子串导致 gold 冗余
        if any(phrase in p or p in phrase for p in out):
            continue
        out.append(phrase)
        if len(out) >= limit:
            break
    return out


def template_questions(source: str, text: str, variants: int = 4) -> list[dict]:
    templates = [
        "《{doc}》中，「{kp}」具体是什么？",
        "原文如何描述「{kp}」？",
        "请说明《{doc}》里「{kp}」的含义。",
        "《{doc}》关于「{kp}」的说明是什么？",
        "根据《{doc}》，「{kp}」指的是什么？",
    ]
    items: list[dict] = []
    for i, kp in enumerate(pick_keyphrases(text, limit=variants)):
        question = templates[i % len(templates)].format(doc=source, kp=kp)
        items.append(
            {
                "question": question,
                "gold": [{"source": source, "must_contain": kp}],
                "refusal_expected": False,
            }
        )
    # 短块可抽取短语不足时，用首行原文补充一题，保证候选规模
    if len(items) < variants and text.strip():
        fallback = text.strip().splitlines()[0][:30]
        used = [i["gold"][0]["must_contain"] for i in items]
        if len(fallback) >= 8 and not any(fallback in u or u in fallback for u in used):
            items.append(
                {
                    "question": f"《{source}》中提到「{fallback}」，具体上下文是什么？",
                    "gold": [{"source": source, "must_contain": fallback}],
                    "refusal_expected": False,
                }
            )
    return items


REFUSAL_TEMPLATES = [
    "《{doc}》的作者是谁？",
    "《{doc}》首次发布于哪一年？",
    "《{doc}》在 GitHub 上有多少 star？",
]


def refusal_items(sources: list[str], limit: int) -> list[dict]:
    """每篇文档可贡献多条不可回答的问题（作者/年份/star 等）。"""
    items: list[dict] = []
    if not sources:
        return items
    i = 0
    while len(items) < limit and i < limit * len(REFUSAL_TEMPLATES):
        source = sources[i % len(sources)]
        tpl = REFUSAL_TEMPLATES[(i // len(sources)) % len(REFUSAL_TEMPLATES)]
        question = tpl.format(doc=source)
        if all(it["question"] != question for it in items):
            items.append(
                {"question": question, "gold": [], "refusal_expected": True}
            )
        i += 1
    return items


def _round_robin(groups: dict[str, list[dict]], limit: int) -> list[dict]:
    out: list[dict] = []
    queues = {k: list(v) for k, v in groups.items()}
    while len(out) < limit and any(queues.values()):
        for key in list(queues):
            if queues[key] and len(out) < limit:
                out.append(queues[key].pop(0))
    return out


def build_template_dataset(
    chunks: list[dict], target: int = 300, variants: int = 4, refusal_ratio: float = 0.08
) -> list[dict]:
    """按文档轮转取样，保证 300 题覆盖全部文档；附带约 8% 拒答题。"""
    n_refusal = max(3, min(target // 4, int(target * refusal_ratio)))
    groups: dict[str, list[dict]] = {}
    seen: set[tuple[str, str]] = set()
    seen_q: set[str] = set()  # 不同文档可能共享原文短语，按问题去重避免重复题
    for chunk in chunks:
        source = chunk.get("source") or "unknown"
        for item in template_questions(source, chunk.get("text", ""), variants=variants):
            key = (source, item["gold"][0]["must_contain"])
            if key in seen or item["question"] in seen_q:
                continue
            seen.add(key)
            seen_q.add(item["question"])
            groups.setdefault(source, []).append(item)

    answerable = _round_robin(groups, target - n_refusal)
    refusals = refusal_items(sorted(groups), n_refusal)
    items = answerable + refusals
    if len(items) < target:  # 文档/块不足时用未使用的答案题补齐（不重复问题）
        seen_q = {i["question"] for i in items}
        for item in answerable:
            if len(items) >= target:
                break
            if item["question"] not in seen_q:
                seen_q.add(item["question"])
                items.append(dict(item))
    random.Random(42).shuffle(items)
    return items[:target]


def load_chunks_from_docs(
    docs_dir: str | Path, chunk_size: int = 500, overlap: int = 80
) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(Path(docs_dir).glob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".markdown"}:
            continue
        source = clean_doc_name(path.name)
        text = path.read_text(encoding="utf-8", errors="ignore")
        for idx, piece in enumerate(recursive_split(text, chunk_size, overlap)):
            if len(piece.strip()) < 40:
                continue
            rows.append(
                {"text": piece, "source": source, "page": 1, "chunk_idx": idx}
            )
    return rows


def build_llm_dataset(
    chunks: list[dict], client, target: int = 300, max_chunks: int | None = None
) -> list[dict]:
    """调用 LLM 生成自然问题；解析失败/短语不在原文时跳过。"""
    items: list[dict] = []
    seen: set[str] = set()
    for chunk in chunks[: max_chunks or len(chunks)]:
        if len(items) >= target:
            break
        prompt = QA_PROMPT.format(source=chunk.get("source", ""), text=chunk.get("text", ""))
        for _ in range(2):  # 结构失败重试 1 次
            try:
                raw = client.chat([{"role": "user", "content": prompt}])
                obj = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
                question = str(obj.get("question", "")).strip()
                kp = str(obj.get("must_contain", "")).strip()
            except Exception:
                continue
            if not question or not kp or kp not in chunk.get("text", ""):
                continue
            if question in seen:
                continue
            seen.add(question)
            items.append(
                {
                    "question": question,
                    "gold": [{"source": chunk.get("source", ""), "must_contain": kp}],
                    "refusal_expected": False,
                }
            )
            break
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="scripts/demo_data", help="文档目录（md/txt）")
    ap.add_argument("--chunks", default=None, help="直接读取 chunks JSONL（含 text/source/page）")
    ap.add_argument("--out", default="evaluation/qa_zh_300.jsonl")
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--mode", choices=["template", "llm"], default="template")
    ap.add_argument("--variants", type=int, default=6, help="每块生成的模板题数量")
    ap.add_argument("--chunk-size", type=int, default=320)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    if args.chunks:
        chunks = [
            json.loads(line)
            for line in Path(args.chunks).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        chunks = load_chunks_from_docs(args.docs, args.chunk_size, args.overlap)

    if args.mode == "llm":
        from app.config import load_config
        from core.llm import LLMClient

        cfg = load_config(args.config)
        items = build_llm_dataset(chunks, LLMClient(cfg["llm"]), target=args.target)
    else:
        items = build_template_dataset(chunks, target=args.target, variants=args.variants)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    n_ref = sum(1 for i in items if i.get("refusal_expected"))
    print(
        f"已生成 {out}：{len(items)} 题（拒答 {n_ref} 题，"
        f"来源块 {len(chunks)}，mode={args.mode}）"
    )


if __name__ == "__main__":
    main()
