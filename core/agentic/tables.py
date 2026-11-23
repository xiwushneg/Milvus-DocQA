"""表格工件：从现有 chunk 文本解析管道表格，落 CSV + registry，DuckDB 只读查询。

硬约束：
- 不新增 Milvus collection / 索引 / 字段，表格只在进程内从 chunk 文本识别；
- CSV 工件写 ``data/tables/``，DuckDB 只查内存视图，不访问用户任意文件；
- ``duckdb`` 是可选依赖，延迟导入；未安装时返回 ``tool_unavailable``，
  Agent 自动回退文本检索，基础服务不受影响。

解析口径（与设计文档 §5.2 一致）：
- 扫描 chunk 文本中的管道行（DOCX 表格已被 loader 压成 ``" | "`` 行）；
- 列数一致且至少 2 行（首行作表头）才建表；
- 列名去重、空列名补 ``colN``。
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TABLES_DIR = "data/tables"
REGISTRY_NAME = "registry.json"

_SEPARATOR = re.compile(r"^:?-{2,}:?$")
_NUMBER = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?$")


# ==================== 解析 ====================
def _split_row(line: str) -> list[str]:
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    return [cell.strip() for cell in text.split("|")]


def _is_pipe_row(line: str) -> bool:
    if "|" not in line:
        return False
    cells = _split_row(line)
    return len(cells) >= 2 and any(cells)


def _is_separator(line: str) -> bool:
    cells = [c for c in _split_row(line) if c]
    return bool(cells) and all(_SEPARATOR.match(c) for c in cells)


def _normalize_columns(header: list[str]) -> list[str]:
    columns: list[str] = []
    seen: dict[str, int] = {}
    for i, name in enumerate(header, 1):
        col = (name or "").strip() or f"col{i}"
        if col in seen:
            seen[col] += 1
            col = f"{col}_{seen[col]}"
        else:
            seen[col] = 1
        columns.append(col)
    return columns


def parse_pipe_table(text: str) -> dict | None:
    """解析文本中的第一个有效管道表格。

    返回 ``{"columns": [...], "rows": [[...]]}``；不满足
    “列数一致且至少 2 行”时返回 None。
    """
    if not text:
        return None
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if _is_pipe_row(line)), None)
    if start is None:
        return None

    header = _split_row(lines[start])
    if len(header) < 2:
        return None
    width = len(header)
    columns = _normalize_columns(header)
    rows: list[list[str]] = []
    for line in lines[start + 1 :]:
        if not line.strip():
            break
        if not _is_pipe_row(line):
            break
        if _is_separator(line):
            continue
        cells = _split_row(line)
        if len(cells) != width:
            break
        rows.append(cells)
    if not rows:
        return None
    return {"columns": columns, "rows": rows}


def to_number(value: str) -> float | None:
    """把单元格转成数值；支持千分位与百分比，失败返回 None。"""
    text = str(value).strip()
    if not text or not _NUMBER.match(text):
        return None
    body = text.replace(",", "")
    try:
        if body.endswith("%"):
            return float(body[:-1]) / 100
        return float(body)
    except ValueError:
        return None


# ==================== 工件与注册表 ====================
@dataclass
class TableArtifact:
    table_id: str
    source: str = ""
    page: int = 0
    chunk_idx: int = -1
    columns: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    csv_path: str = ""

    @property
    def numeric_columns(self) -> list[str]:
        out: list[str] = []
        for i, col in enumerate(self.columns):
            vals = [r[i] for r in self.rows if i < len(r) and str(r[i]).strip()]
            if vals and all(to_number(v) is not None for v in vals):
                out.append(col)
        return out

    def column_types(self) -> list[str]:
        numeric = set(self.numeric_columns)
        return ["DOUBLE" if c in numeric else "VARCHAR" for c in self.columns]

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "source": self.source,
            "page": self.page,
            "chunk_idx": self.chunk_idx,
            "columns": list(self.columns),
            "rows": [list(r) for r in self.rows],
            "csv_path": self.csv_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TableArtifact":
        return cls(
            table_id=str(data["table_id"]),
            source=str(data.get("source", "")),
            page=int(data.get("page", 0)),
            chunk_idx=int(data.get("chunk_idx", -1)),
            columns=[str(c) for c in data.get("columns", [])],
            rows=[[str(c) for c in row] for row in data.get("rows", [])],
            csv_path=str(data.get("csv_path", "")),
        )


class TableRegistry:
    """进程内表格注册表；可选持久化到 ``data/tables/registry.json``。"""

    def __init__(self, tables: list[TableArtifact] | None = None):
        self._by_id: dict[str, TableArtifact] = {}
        for table in tables or []:
            self.add(table)

    def add(self, table: TableArtifact) -> None:
        self._by_id[table.table_id] = table

    def get(self, table_id: str) -> TableArtifact | None:
        return self._by_id.get(table_id)

    def all(self) -> list[TableArtifact]:
        return list(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    @classmethod
    def build(
        cls,
        rows: list[dict],
        out_dir: str | Path | None = None,
        *,
        max_tables: int = 500,
    ) -> "TableRegistry":
        """从 Milvus 全量 chunk 构建表格工件。

        ``rows`` 需含 id/text/source/page/chunk_idx（缺 id 时用序号兜底）。
        """
        tables: list[TableArtifact] = []
        used_ids: set[str] = set()
        for i, row in enumerate(rows):
            text = str(row.get("text", ""))
            parsed = parse_pipe_table(text)
            if parsed is None:
                continue
            chunk_id = row.get("id", i + 1)
            table_id = f"t{chunk_id}"
            if table_id in used_ids:
                table_id = f"{table_id}_{len(used_ids)}"
            used_ids.add(table_id)
            tables.append(
                TableArtifact(
                    table_id=table_id,
                    source=str(row.get("source", "")),
                    page=int(row.get("page", 0)),
                    chunk_idx=int(row.get("chunk_idx", -1)),
                    columns=parsed["columns"],
                    rows=parsed["rows"],
                )
            )
            if len(tables) >= max_tables:
                break

        registry = cls(tables)
        if out_dir is None:
            return registry
        try:
            registry.save(out_dir)
        except OSError:
            # 工件落盘失败不影响内存查询；trace 由调用方记录
            pass
        return registry

    def ensure_built(
        self, rows: list[dict], out_dir: str | Path | None = None, *, max_tables: int = 500
    ) -> "TableRegistry":
        """懒加载兜底：注册表为空时才从全量 chunk 构建。"""
        if self._by_id or not rows:
            return self
        built = TableRegistry.build(rows, out_dir, max_tables=max_tables)
        self._by_id = built._by_id
        return self

    def find(self, query: str, *, limit: int = 3) -> list[TableArtifact]:
        """按 source / 列名 / 单元格内容与问题的词重叠度排序。"""
        from core.bm25 import tokenize

        q = (query or "").lower()
        q_tokens = set(tokenize(q))
        scored: list[tuple[float, TableArtifact]] = []
        for table in self._by_id.values():
            score = 0.0
            if table.source and table.source.lower() in q:
                score += 5.0
            for col in table.columns:
                if col.lower() in q:
                    score += 3.0
                score += len(set(tokenize(col.lower())) & q_tokens)
            for row in table.rows[:20]:
                for cell in row:
                    if cell and cell.lower() in q:
                        score += 1.0
            scored.append((score, table))

        scored.sort(key=lambda item: item[0], reverse=True)
        hit = [t for s, t in scored if s > 0][:limit]
        if not hit and len(scored) == 1:
            # 全库只有一张表时允许兜底，避免表头与问题措辞差异导致漏检
            hit = [scored[0][1]]
        return hit

    # ---------- 持久化 ----------
    def save(self, out_dir: str | Path) -> Path:
        base = Path(out_dir)
        base.mkdir(parents=True, exist_ok=True)
        for table in self._by_id.values():
            digest = hashlib.sha1(table.source.encode("utf-8")).hexdigest()[:8]
            name = f"{digest}_{table.chunk_idx}.csv"
            path = base / name
            with path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(table.columns)
                writer.writerows(table.rows)
            table.csv_path = str(path)

        payload = {
            "version": 1,
            "tables": [t.to_dict() for t in self._by_id.values()],
        }
        target = base / REGISTRY_NAME
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, out_dir: str | Path) -> "TableRegistry":
        path = Path(out_dir) / REGISTRY_NAME
        if not path.is_file():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        if not isinstance(payload, dict):
            return cls()
        tables = []
        for item in payload.get("tables", []):
            try:
                tables.append(TableArtifact.from_dict(item))
            except (KeyError, TypeError, ValueError):
                continue  # 单条损坏不阻塞其余表加载
        return cls(tables)


# ==================== DuckDB 只读查询 ====================
_FORBIDDEN_SQL = re.compile(
    r"\b(attach|detach|copy|install|load|pragma|export|import|call|create|drop|alter"
    r"|insert|update|delete|truncate|vacuum|checkpoint|use|set|reset"
    r"|read_csv|read_csv_auto|read_parquet|read_json|read_json_auto|read_text"
    r"|read_blob|glob|sniff_csv)\b",
    re.IGNORECASE,
)
_TABLE_REF = re.compile(r"\b(?:from|join)\s+(\"[^\"]+\"|[A-Za-z_][\w]*)", re.IGNORECASE)
_CTE_NAME = re.compile(r"(?:\bwith\b|,)\s*([A-Za-z_][\w]*)\s+as\s*\(", re.IGNORECASE)
_COMMENT = re.compile(r"--|/\*")
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")


def _mask_literals(sql: str) -> str:
    """把单引号字符串字面量替换为空串，用于关键字/表名检查。

    字面量里的 ``set``、``load``、``from`` 等词不应触发只读校验；
    ``''`` 的两段引号保留，避免影响相邻文本的边界判断。
    """
    return _STRING_LITERAL.sub("''", sql)


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def validate_readonly_sql(sql: str, allowed_tables: set[str]) -> str:
    """校验只读 SQL；不合法时抛出 ValueError(code)。

    校验前先屏蔽单引号字面量，避免 ``SELECT 'set load' AS label``
    这类正常查询被关键字黑名单误拒；返回原始（去尾分号）SQL。
    """
    text = (sql or "").strip()
    if not text:
        raise ValueError("empty_sql")
    if len(text) > 2000:
        raise ValueError("sql_too_long")
    if text.endswith(";"):
        text = text[:-1].strip()
    masked = _mask_literals(text)
    if _COMMENT.search(masked):
        raise ValueError("comments_not_allowed")
    if ";" in masked:
        raise ValueError("multi_statement")
    if _FORBIDDEN_SQL.search(masked):
        raise ValueError("unsafe_sql")
    if not re.match(r"^(select|with)\b", masked, re.IGNORECASE):
        raise ValueError("only_select_allowed")

    ctes = {name.lower() for name in _CTE_NAME.findall(masked)}
    for raw in _TABLE_REF.findall(masked):
        name = raw.strip('"')
        if name.lower() not in allowed_tables and name.lower() not in ctes:
            raise ValueError("table_not_allowed")
    return text


class DuckDBTool:
    """对注册表内表格执行只读 SQL（内存视图，行数上限 + 超时中断）。"""

    def __init__(
        self,
        registry: TableRegistry,
        *,
        max_rows: int = 200,
        timeout_ms: int = 3000,
    ):
        self.registry = registry
        self.max_rows = max(1, int(max_rows))
        self.timeout_ms = max(100, int(timeout_ms))
        # 表只从 chunk 文本构建一次：缓存内存连接，避免每次查询重灌全部表
        self._con = None
        self._con_stamp: tuple[str, ...] | None = None
        self._con_lock = threading.Lock()
        self._exec_lock = threading.Lock()

    def _ensure_connection(self):
        """返回已装载全部表的缓存连接；注册表变化时重建。"""
        import duckdb  # 延迟导入：可选依赖

        tables = self.registry.all()
        stamp = tuple(t.table_id for t in tables)
        with self._con_lock:
            if self._con is not None and self._con_stamp == stamp:
                return self._con
            self._close_connection_locked()
            con = duckdb.connect(database=":memory:")
            try:
                for table in tables:
                    self._create_table(con, table)
            except BaseException:
                con.close()
                raise
            self._con, self._con_stamp = con, stamp
            return con

    def _close_connection_locked(self) -> None:
        con, self._con = self._con, None
        self._con_stamp = None
        if con is not None:
            try:
                con.close()
            except Exception:
                pass

    def _reset_connection(self) -> None:
        with self._con_lock:
            self._close_connection_locked()

    def run(self, sql: str, table_id: str = "") -> dict:
        t0 = time.perf_counter()
        result: dict = {
            "ok": False,
            "sql": sql,
            "table_id": table_id,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "latency_ms": 0,
            "error": "",
        }
        tables = self.registry.all()
        allowed = {t.table_id.lower() for t in tables}
        if table_id and table_id.lower() not in allowed:
            result["error"] = "table_not_allowed"
            result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
            return result
        if not tables:
            result["error"] = "no_table"
            result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
            return result

        try:
            safe_sql = validate_readonly_sql(sql, allowed)
        except ValueError as exc:
            result["error"] = str(exc)
            result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
            return result

        box: dict = {}

        def worker() -> None:
            try:
                with self._exec_lock:  # DuckDB 连接不支持并发执行
                    con = self._ensure_connection()
                    box["con"] = con
                    cur = con.execute(safe_sql)
                    box["columns"] = [d[0] for d in (cur.description or [])]
                    box["rows"] = cur.fetchmany(self.max_rows + 1)
            except BaseException as exc:  # 线程内异常统一回传
                box["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(self.timeout_ms / 1000)
        if thread.is_alive():
            con = box.get("con")
            if con is not None:
                try:
                    con.interrupt()
                except Exception:
                    pass
            thread.join(0.2)
            # 中断后的连接状态不可信，丢弃缓存让下次查询重建
            self._reset_connection()
            result["error"] = "timeout"
            result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
            return result

        err = box.get("error")
        if err is not None:
            if isinstance(err, ImportError):
                result["error"] = "tool_unavailable"
            else:
                result["error"] = f"query_error: {err}"
            result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
            return result

        rows = box.get("rows") or []
        truncated = len(rows) > self.max_rows
        rows = rows[: self.max_rows]
        result.update(
            ok=True,
            columns=[str(c) for c in box.get("columns", [])],
            rows=[[_json_cell(c) for c in row] for row in rows],
            row_count=len(rows),
            truncated=truncated,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
        return result

    @staticmethod
    def _create_table(con, table: TableArtifact) -> None:
        col_defs = ", ".join(
            f"{quote_ident(c)} {t}"
            for c, t in zip(table.columns, table.column_types())
        )
        con.execute(f"CREATE TABLE {quote_ident(table.table_id)} ({col_defs})")
        if not table.rows:
            return
        numeric = set(table.numeric_columns)
        values: list[tuple] = []
        for row in table.rows:
            cells = []
            for col, cell in zip(table.columns, row):
                if col in numeric:
                    num = to_number(cell)
                    cells.append(num if num is not None else None)
                else:
                    cells.append(str(cell))
            values.append(tuple(cells))
        placeholders = ", ".join("?" for _ in table.columns)
        con.executemany(
            f"INSERT INTO {quote_ident(table.table_id)} VALUES ({placeholders})",
            values,
        )


def _json_cell(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
