"""块类型内存判定：prose / table / code / enum / formula。

设计约束：不新增 Milvus collection、索引或字段；类型只在进程内按规则计算并
缓存，数据源为 BM25Index._meta（已有全量 chunk）。规则确定性、可解释，
不使用模型，便于答辩与复现。
"""
from __future__ import annotations

import re
from collections.abc import Callable

# 表格：至少 2 行、列数一致
_MIN_TABLE_ROWS = 2
# 代码：围栏优先；否则要求长度 + 符号密度 + 代码关键字三重证据
_CODE_MIN_LEN = 60
_CODE_SYMBOL_DENSITY = 0.06
_CODE_KEYWORD = re.compile(
    r"\b(def|class|import|return|function|const|let|var|SELECT|INSERT|UPDATE"
    r"|curl|pip\s+install|npm\s+install)\b"
)
_ENUM_LINE = re.compile(
    r"^\s*(?:[-*•]|\d+[.)、]|\(\d+\)|（\d+）|[①-⑩])\s+\S"
)
_FORMULA_MARK = re.compile(r"[=Σ∑∫√≤≥±×÷]")


def _table_columns(line: str) -> int:
    """返回一行的列数；不是表格行时返回 0。"""
    stripped = line.strip()
    if "|" in stripped:
        cols = [c for c in stripped.strip("|").split("|")]
        return len(cols) if len(cols) >= 2 else 0
    if "\t" in stripped:
        cols = stripped.split("\t")
        return len(cols) if len(cols) >= 2 else 0
    return 0


def _has_table(text: str) -> bool:
    run: list[int] = []
    for line in text.splitlines():
        cols = _table_columns(line)
        if cols <= 0:
            run = []
            continue
        run.append(cols)
        if len(run) < _MIN_TABLE_ROWS:
            continue
        if len(set(run)) == 1:
            return True
    return False


def _has_code(text: str) -> bool:
    if text.count("```") >= 2:
        return True
    body = text.strip()
    if len(body) < _CODE_MIN_LEN:
        return False
    if not _CODE_KEYWORD.search(body):
        return False
    symbols = sum(1 for ch in body if ch in "{}[]();:=<>/\\_*#")
    return symbols / len(body) >= _CODE_SYMBOL_DENSITY


def _has_enum(text: str) -> bool:
    count = sum(1 for line in text.splitlines() if _ENUM_LINE.match(line))
    return count >= 3


def _has_formula(text: str) -> bool:
    if not _FORMULA_MARK.search(text):
        return False
    digits = sum(1 for ch in text if ch.isdigit())
    return digits >= 3 and digits / max(1, len(text)) >= 0.05


def classify(text: str) -> str:
    """按 table → code → enum → formula → prose 的优先级判定块类型。"""
    body = text or ""
    if _has_table(body):
        return "table"
    if _has_code(body):
        return "code"
    if _has_enum(body):
        return "enum"
    if _has_formula(body):
        return "formula"
    return "prose"


class ChunkTypeIndex:
    """chunk_id → 块类型缓存。

    rows_provider 返回全量 chunk（通常为 BM25Index._meta）。提供者数据被重建时
    （对象 id 或行数变化）自动重建缓存；不落库、不改 schema。
    """

    def __init__(self, rows_provider: Callable[[], list[dict]]):
        self._provider = rows_provider
        self._cache: dict[int, str] = {}
        self._key: tuple[int, int] | None = None

    def _ensure(self) -> None:
        rows = self._provider() or []
        key = (id(rows), len(rows))
        if key == self._key:
            return
        cache: dict[int, str] = {}
        for row in rows:
            try:
                cid = int(row["id"])
            except (KeyError, TypeError, ValueError):
                continue
            cache[cid] = classify(str(row.get("text", "")))
        self._cache = cache
        self._key = key

    def get(self, chunk_id: int, text: str = "") -> str:
        """优先取缓存；新块（尚未进入 BM25）按文本即时判定。"""
        self._ensure()
        try:
            cid = int(chunk_id)
        except (TypeError, ValueError):
            return classify(text)
        return self._cache.get(cid) or classify(text)

    def annotate(
        self,
        hits: list[dict],
        *,
        id_key: str = "id",
        text_key: str = "text",
    ) -> list[dict]:
        """就地写入 chunk_type 并返回 hits，便于链式调用。"""
        for hit in hits:
            hit["chunk_type"] = self.get(hit.get(id_key, -1), str(hit.get(text_key, "")))
        return hits

    def stats(self) -> dict[str, int]:
        self._ensure()
        out: dict[str, int] = {}
        for ctype in self._cache.values():
            out[ctype] = out.get(ctype, 0) + 1
        return out


def count_types(hits: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for hit in hits:
        ctype = str(hit.get("chunk_type") or "prose")
        out[ctype] = out.get(ctype, 0) + 1
    return out


__all__ = ["ChunkTypeIndex", "classify", "count_types"]
