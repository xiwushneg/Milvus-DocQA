"""文本分块模块：递归字符分割（中文标点感知 + 相邻块重叠）。

设计要点（对应论文 3.x 节）：
1. 分隔符按粒度从粗到细依次尝试：空行→换行→中文句号/问叹号→英文句读→空格→硬切；
2. 超长且无分隔符的文本按滑窗硬切（步长 = chunk_size - overlap），保证无遗漏；
3. 相邻块保留 chunk_overlap 字符重叠，避免关键信息恰好被块边界切断；
4. 过短尾块（< min_chunk_size）并入前一块，避免碎片噪声。
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", " ", ""]


@dataclass
class Chunk:
    text: str
    source: str = ""
    page: int = 0
    chunk_idx: int = 0
    start: int = -1
    end: int = -1

    def to_row(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "chunk_idx": self.chunk_idx,
        }


def recursive_split(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    separators: list[str] | None = None,
) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    seps = list(separators) if separators else list(DEFAULT_SEPARATORS)

    def hard_split(t: str) -> list[str]:
        step = chunk_size - chunk_overlap
        return [t[i : i + chunk_size] for i in range(0, len(t), step)] if t else []

    def split_by(t: str, sep: str) -> list[str]:
        """按分隔符切分并回贴分隔符，保证原文无损。"""
        parts = t.split(sep)
        out: list[str] = []
        for i, p in enumerate(parts):
            if i < len(parts) - 1:
                out.append(p + sep)
            elif p:
                out.append(p)
        return out

    def rec(t: str, si: int) -> list[str]:
        t = t.strip("\n")
        if not t:
            return []
        if len(t) <= chunk_size:
            return [t]
        if si >= len(seps):
            return hard_split(t)
        sep = seps[si]
        if sep and sep not in t:
            return rec(t, si + 1)
        parts = hard_split(t) if sep == "" else split_by(t, sep)
        chunks: list[str] = []
        buf = ""
        for p in parts:
            if not p.strip():
                continue
            if len(p) > chunk_size:  # 单段超长 → 继续细粒度递归
                if buf.strip():
                    chunks.append(buf)
                    buf = ""
                chunks.extend(rec(p, si + 1))
                continue
            if len(buf) + len(p) <= chunk_size:
                buf += p
            else:
                if buf.strip():
                    chunks.append(buf)
                tail = buf[-chunk_overlap:] if chunk_overlap and len(buf) > chunk_overlap else ""
                buf = (tail + p) if len(tail) + len(p) <= chunk_size else p
        if buf.strip():
            chunks.append(buf)
        return chunks

    return rec(text, 0)


def split_segments(
    segments: list[dict],
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    min_chunk_size: int = 30,
) -> list[Chunk]:
    """将解析出的段落列表分块，返回带溯源元数据的 Chunk 列表。"""
    chunks: list[Chunk] = []
    for seg in segments:
        for piece in recursive_split(seg["text"], chunk_size, chunk_overlap):
            chunks.append(
                Chunk(text=piece, source=source, page=int(seg.get("page", 0)))
            )

    merged: list[Chunk] = []
    for c in chunks:
        if (
            merged
            and len(c.text) < min_chunk_size
            and len(merged[-1].text) + len(c.text) <= chunk_size + min_chunk_size
        ):
            merged[-1].text += c.text
        else:
            merged.append(c)
    for i, c in enumerate(merged):
        c.chunk_idx = i
    return merged
