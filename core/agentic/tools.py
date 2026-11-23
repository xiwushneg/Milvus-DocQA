"""工具层：DuckDB 查表、计算器、单位换算、代码执行与工具规划。

安全约束（设计 §5.4）：
- ``calculator``：AST 白名单，禁止属性/下标/导入/非白名单调用；
- ``unit_convert``：预置单位表，未知单位直接报错；
- ``code_exec``：子进程 + 超时 + 空环境变量，**默认关闭**，仅本地演示；
- ``duckdb_query``：只读 SQL、表名白名单、行数上限与超时（见 tables.py）。

调用方式：brain 支持 tools 时走 ``chat_with_tools``；否则按 Analyze 输出的
``needs_table / needs_calculation`` 做确定性规划，最多 ``max_tool_calls`` 次。
"""
from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from core.agentic.tables import DuckDBTool, TableArtifact, TableRegistry, quote_ident

# ==================== 结果对象 ====================
@dataclass
class ToolResult:
    name: str
    ok: bool
    output: str = ""
    data: dict = field(default_factory=dict)
    error: str = ""
    arguments: dict = field(default_factory=dict)
    latency_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "tool": self.name,
            "ok": self.ok,
            "output": self.output,
            "data": self.data,
            "error": self.error,
            "arguments": self.arguments,
            "latency_ms": self.latency_ms,
            "provenance": "tool",
        }


# ==================== 计算器（AST 白名单） ====================
_MAX_EXPR_LEN = 200
_MAX_POW = 100
_MAX_RESULT = 1e15
_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("unsafe_constant")
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_eval_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return _eval_node(node.operand)
        raise ValueError("unsafe_unary_op")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError("unsafe_operator")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW:
            raise ValueError("exponent_too_large")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("division_by_zero")
        value = op(left, right)
        if not math.isfinite(value) or abs(value) > _MAX_RESULT:
            raise ValueError("result_out_of_range")
        return value
    raise ValueError("unsafe_expression")


def evaluate_expression(expression: str) -> float:
    """求值算术表达式；仅允许数字、括号与四则/幂/取模运算。"""
    text = (expression or "").strip()
    if not text:
        raise ValueError("empty_expression")
    if len(text) > _MAX_EXPR_LEN:
        raise ValueError("expression_too_long")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid_expression") from exc
    return _eval_node(tree)


_CN_OPS = {
    "乘以": "*",
    "除以": "/",
    "加上": "+",
    "减去": "-",
    "×": "*",
    "÷": "/",
    "＋": "+",
    "－": "-",
    "＊": "*",
    "／": "/",
}
_EXPR_RE = re.compile(
    r"[-+]?\d+(?:\.\d+)?(?:\s*(?:\*\*|[+\-*/%])\s*[-+]?\d+(?:\.\d+)?)+"
)


def extract_expression(text: str) -> str:
    """从中文问题里抽取算术表达式（如 “12.5 * 3 + 7”）。"""
    body = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text or "")
    for word, symbol in _CN_OPS.items():
        body = body.replace(word, symbol)
    match = _EXPR_RE.search(body)
    return match.group(0).strip() if match else ""


def _fmt_number(value: float) -> str:
    if abs(value - round(value)) < 1e-12 and abs(value) < 1e15:
        return str(int(round(value)))
    return f"{value:.10g}"


# ==================== 单位换算 ====================
# name -> (category, factor to base)
_UNITS: dict[str, tuple[str, float]] = {
    # 长度（基准：米）
    "m": ("length", 1.0), "米": ("length", 1.0),
    "km": ("length", 1000.0), "千米": ("length", 1000.0), "公里": ("length", 1000.0),
    "cm": ("length", 0.01), "厘米": ("length", 0.01),
    "mm": ("length", 0.001), "毫米": ("length", 0.001),
    "mile": ("length", 1609.344), "miles": ("length", 1609.344), "英里": ("length", 1609.344),
    "ft": ("length", 0.3048), "feet": ("length", 0.3048), "英尺": ("length", 0.3048),
    "inch": ("length", 0.0254), "in": ("length", 0.0254), "英寸": ("length", 0.0254),
    "yard": ("length", 0.9144), "码": ("length", 0.9144),
    # 质量（基准：千克）
    "kg": ("mass", 1.0), "千克": ("mass", 1.0), "公斤": ("mass", 1.0),
    "g": ("mass", 0.001), "克": ("mass", 0.001),
    "mg": ("mass", 1e-6), "毫克": ("mass", 1e-6),
    "t": ("mass", 1000.0), "吨": ("mass", 1000.0),
    "lb": ("mass", 0.45359237), "lbs": ("mass", 0.45359237), "磅": ("mass", 0.45359237),
    "斤": ("mass", 0.5),
    # 数据量（基准：字节）
    "b": ("data", 1.0), "byte": ("data", 1.0), "字节": ("data", 1.0),
    "kb": ("data", 1024.0), "mb": ("data", 1024.0**2),
    "gb": ("data", 1024.0**3), "tb": ("data", 1024.0**4),
    # 时间（基准：秒）
    "s": ("time", 1.0), "秒": ("time", 1.0),
    "min": ("time", 60.0), "分钟": ("time", 60.0),
    "h": ("time", 3600.0), "小时": ("time", 3600.0),
    "day": ("time", 86400.0), "天": ("time", 86400.0), "日": ("time", 86400.0),
    # 温度（单独处理）
    "c": ("temperature", 1.0), "℃": ("temperature", 1.0), "摄氏度": ("temperature", 1.0),
    "f": ("temperature", 1.0), "℉": ("temperature", 1.0), "华氏度": ("temperature", 1.0),
    "k": ("temperature", 1.0), "开尔文": ("temperature", 1.0), "开": ("temperature", 1.0),
}


def _normalize_unit(unit: str) -> str:
    text = (unit or "").strip().lower()
    return text.rstrip("。.，,；;!！?？")


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """单位换算；未知单位或跨类别时抛 ValueError。"""
    src = _normalize_unit(from_unit)
    dst = _normalize_unit(to_unit)
    if src not in _UNITS:
        raise ValueError(f"unknown_unit: {from_unit}")
    if dst not in _UNITS:
        raise ValueError(f"unknown_unit: {to_unit}")
    src_cat, src_factor = _UNITS[src]
    dst_cat, dst_factor = _UNITS[dst]
    if src_cat != dst_cat:
        raise ValueError("unit_mismatch")

    num = float(value)
    if src_cat == "temperature":
        if src == "f" or src == "℉" or src == "华氏度":
            celsius = (num - 32.0) * 5.0 / 9.0
        elif src == "k" or src == "开尔文" or src == "开":
            celsius = num - 273.15
        else:
            celsius = num
        if dst == "f" or dst == "℉" or dst == "华氏度":
            return celsius * 9.0 / 5.0 + 32.0
        if dst == "k" or dst == "开尔文" or dst == "开":
            return celsius + 273.15
        return celsius
    return num * src_factor / dst_factor


_CONVERT_RE = re.compile(
    r"(?P<value>-?\d+(?:\.\d+)?)\s*"
    r"(?P<from>[^\s\d，。？,、:：]{1,6}?)\s*"
    r"(?:换算成|换算为|转换成|转换为|转成|转为|等于多少|是多少|等于|折合|to|in|=)\s*"
    r"(?P<to>[^\s\d，。？,、:：]{1,6})"
)


def extract_conversion(text: str) -> dict | None:
    """抽取 “100 公里等于多少英里” 这类换算三要素。"""
    match = _CONVERT_RE.search(text or "")
    if not match:
        return None
    frm = _normalize_unit(match.group("from")).replace("的", "")
    to = _normalize_unit(match.group("to")).replace("的", "")
    if frm not in _UNITS or to not in _UNITS:
        return None
    try:
        value = float(match.group("value"))
    except ValueError:
        return None
    return {"value": value, "from_unit": frm, "to_unit": to}


# ==================== 代码执行 ====================
def run_python(code: str, timeout_ms: int = 3000) -> dict:
    """子进程执行 Python 片段；Windows 无强沙箱，仅本地演示。"""
    if not (code or "").strip():
        raise ValueError("empty_code")
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=max(0.1, timeout_ms / 1000),
            env={},
            cwd=tempfile.gettempdir(),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "", "error": "timeout", "data": {}}
    stdout = (proc.stdout or "").strip()[:4000]
    stderr = (proc.stderr or "").strip()[:4000]
    ok = proc.returncode == 0
    output = stdout if ok else "\n".join(p for p in (stdout, stderr) if p)
    return {
        "ok": ok,
        "output": output,
        "error": "" if ok else "execution_failed",
        "data": {"returncode": proc.returncode, "stdout": stdout, "stderr": stderr},
    }


# ==================== 工具注册表 ====================
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., dict]
    enabled: bool = True

    def spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        tool = self._tools.get(name)
        return bool(tool and tool.enabled)

    def names(self) -> list[str]:
        return [n for n, t in self._tools.items() if t.enabled]

    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values() if t.enabled]

    def execute(self, name: str, arguments: dict | str | None = None) -> ToolResult:
        t0 = time.perf_counter()
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name=name, ok=False, error="tool_not_found")
        if not tool.enabled:
            return ToolResult(name=name, ok=False, error="tool_disabled")

        args = arguments
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                return ToolResult(name=name, ok=False, error="bad_arguments")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return ToolResult(name=name, ok=False, error="bad_arguments")

        try:
            raw = tool.fn(**args)
        except TypeError as exc:
            return ToolResult(
                name=name,
                ok=False,
                error="bad_arguments",
                output=str(exc),
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
        except ValueError as exc:
            return ToolResult(
                name=name,
                ok=False,
                error=str(exc),
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
        except Exception as exc:  # 工具内部错误不应中断 Agent
            return ToolResult(
                name=name,
                ok=False,
                error=f"tool_error: {exc}",
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )

        if not isinstance(raw, dict):
            raw = {"ok": True, "output": str(raw)}
        return ToolResult(
            name=name,
            ok=bool(raw.get("ok", True)),
            output=str(raw.get("output", "")),
            data=raw.get("data") or {},
            error=str(raw.get("error", "")),
            arguments=args,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )


def _format_query_result(result: dict) -> str:
    columns = result.get("columns") or []
    rows = result.get("rows") or []
    if not rows:
        return "查询无结果"
    if len(rows) == 1 and len(columns) == 1:
        return f"{columns[0]} = {rows[0][0]}"
    header = " | ".join(str(c) for c in columns)
    body = "\n".join(" | ".join(str(c) for c in row) for row in rows[:20])
    text = f"{header}\n{body}"
    if result.get("truncated") or len(rows) > 20:
        text += "\n（结果已截断）"
    return text


def build_tool_registry(
    tools_cfg: dict | None,
    tables: TableRegistry,
    *,
    duckdb_tool: DuckDBTool | None = None,
) -> ToolRegistry:
    """按配置构建 4 个工具；code_exec 默认关闭。"""
    cfg = tools_cfg or {}
    duck_cfg = cfg.get("duckdb", {}) or {}
    duck = duckdb_tool or DuckDBTool(
        tables,
        max_rows=int(duck_cfg.get("max_rows", 200)),
        timeout_ms=int(duck_cfg.get("timeout_ms", 3000)),
    )

    def duck_fn(sql: str, table_id: str = "") -> dict:
        result = duck.run(sql, table_id)
        if not result.get("ok"):
            return {"ok": False, "error": result.get("error", "query_failed"), "data": result}
        return {"ok": True, "output": _format_query_result(result), "data": result}

    def calc_fn(expression: str) -> dict:
        value = evaluate_expression(expression)
        return {
            "ok": True,
            "output": f"{expression.strip()} = {_fmt_number(value)}",
            "data": {"value": value, "expression": expression},
        }

    def unit_fn(value: float, from_unit: str, to_unit: str) -> dict:
        result = convert_units(float(value), from_unit, to_unit)
        return {
            "ok": True,
            "output": f"{_fmt_number(float(value))} {from_unit} = {_fmt_number(result)} {to_unit}",
            "data": {"value": result, "from_unit": from_unit, "to_unit": to_unit},
        }

    code_cfg = cfg.get("code_exec", {}) or {}

    def code_fn(code: str) -> dict:
        return run_python(code, timeout_ms=int(code_cfg.get("timeout_ms", 3000)))

    return ToolRegistry(
        [
            Tool(
                name="duckdb_query",
                description=(
                    "对白名单表格执行只读 SQL（仅 SELECT/WITH），用于精确聚合、"
                    "比较与极值。表名必须使用给定的 table_id。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "只读 SELECT/WITH 查询"},
                        "table_id": {"type": "string", "description": "目标表格 id，如 t12"},
                    },
                    "required": ["sql"],
                },
                fn=duck_fn,
                enabled=bool(duck_cfg.get("enabled", True)),
            ),
            Tool(
                name="calculator",
                description="计算算术表达式（支持 + - * / 幂 与括号）。",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "如 (12.5*3+7)"}
                    },
                    "required": ["expression"],
                },
                fn=calc_fn,
                enabled=bool((cfg.get("calculator", {}) or {}).get("enabled", True)),
            ),
            Tool(
                name="unit_convert",
                description="数值单位换算（长度/质量/数据量/时间/温度）。",
                parameters={
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "from_unit": {"type": "string"},
                        "to_unit": {"type": "string"},
                    },
                    "required": ["value", "from_unit", "to_unit"],
                },
                fn=unit_fn,
                enabled=bool((cfg.get("unit_convert", {}) or {}).get("enabled", True)),
            ),
            Tool(
                name="code_exec",
                description="执行 Python 片段（仅本地演示，默认关闭）。",
                parameters={
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
                fn=code_fn,
                enabled=bool(code_cfg.get("enabled", False)),
            ),
        ]
    )


# ==================== 工具规划 ====================
@dataclass
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)
    reason: str = "rules"

    def to_dict(self) -> dict:
        return {"name": self.name, "arguments": self.arguments, "reason": self.reason}


_AGG_RULES = [
    ("SUM", re.compile(r"总和|总计|合计|求和|总额|加起来|累计")),
    ("AVG", re.compile(r"平均|均值")),
    ("MAX", re.compile(r"最大|最高|最多")),
    ("MIN", re.compile(r"最小|最低|最少")),
]
_COUNT_RE = re.compile(r"多少行|几行|几条|多少条|多少个|行数|数量是多少|记录数")


def _match_column(question: str, table: TableArtifact, *, numeric_only: bool) -> str | None:
    from core.bm25 import tokenize

    q = question.lower()
    columns = table.numeric_columns if numeric_only else table.columns
    for col in columns:
        if col.lower() in q:
            return col
    q_tokens = set(tokenize(q))
    best: tuple[int, str | None] = (0, None)
    for col in columns:
        overlap = len(set(tokenize(col.lower())) & q_tokens)
        if overlap > best[0]:
            best = (overlap, col)
    if best[1]:
        return best[1]
    if numeric_only and len(table.numeric_columns) == 1:
        return table.numeric_columns[0]
    return None


def build_table_sql(question: str, table: TableArtifact) -> str:
    """把中文聚合意图翻译成只读 SQL；无合适列时退化为预览。"""
    q = question or ""
    if _COUNT_RE.search(q):
        return f'SELECT COUNT(*) AS "行数" FROM {quote_ident(table.table_id)}'
    for func, pattern in _AGG_RULES:
        if not pattern.search(q):
            continue
        col = _match_column(q, table, numeric_only=True)
        if col:
            return (
                f'SELECT {func}({quote_ident(col)}) AS "结果" '
                f"FROM {quote_ident(table.table_id)}"
            )
        break
    return f"SELECT * FROM {quote_ident(table.table_id)}"


class ToolPlanner:
    """选择并参数化工具调用：优先 brain function calling，失败回退确定性规则。"""

    def __init__(
        self,
        registry: ToolRegistry,
        tables: TableRegistry,
        *,
        brain=None,
        max_calls: int = 3,
    ):
        self.registry = registry
        self.tables = tables
        self.brain = brain
        self.max_calls = max(1, int(max_calls))

    # ---------- 对外 ----------
    def plan(
        self,
        question: str,
        *,
        needs_table: bool = False,
        needs_calculation: bool = False,
    ) -> tuple[list[ToolCall], dict | None]:
        trace: dict | None = None
        calls: list[ToolCall] = []
        if self._llm_ready():
            try:
                calls, trace = self._select_with_llm(question)
            except Exception as exc:  # 模型不支持 tools / 网络失败 → 规则兜底
                trace = {
                    "messages": [],
                    "completion": "",
                    "latency_ms": 0,
                    "error": str(exc),
                }
                calls = []
        if not calls:
            calls = self.plan_rules(
                question,
                needs_table=needs_table,
                needs_calculation=needs_calculation,
            )
        return calls[: self.max_calls], trace

    def plan_rules(
        self,
        question: str,
        *,
        needs_table: bool = False,
        needs_calculation: bool = False,
    ) -> list[ToolCall]:
        calls: list[ToolCall] = []
        conversion = extract_conversion(question)
        if conversion and self.registry.has("unit_convert"):
            calls.append(ToolCall("unit_convert", conversion, "unit_rule"))

        if needs_table and self.registry.has("duckdb_query"):
            candidates = self.tables.find(question)
            if candidates:
                table = candidates[0]
                calls.append(
                    ToolCall(
                        "duckdb_query",
                        {"sql": build_table_sql(question, table), "table_id": table.table_id},
                        "table_rule",
                    )
                )

        expression = extract_expression(question)
        if needs_calculation and expression and self.registry.has("calculator"):
            calls.append(
                ToolCall("calculator", {"expression": expression}, "calc_rule")
            )
        return self._dedupe(calls)

    # ---------- LLM 选择 ----------
    def _llm_ready(self) -> bool:
        return (
            self.brain is not None
            and callable(getattr(self.brain, "chat_with_tools", None))
            and bool(self.registry.specs())
        )

    def _select_with_llm(self, question: str) -> tuple[list[ToolCall], dict]:
        messages = self._tool_messages(question)
        t0 = time.perf_counter()
        response = self.brain.chat_with_tools(messages, self.registry.specs())
        latency_ms = int((time.perf_counter() - t0) * 1000)
        raw_calls = (response or {}).get("tool_calls") or []
        calls: list[ToolCall] = []
        for item in raw_calls:
            fn = (item or {}).get("function") or {}
            name = str(fn.get("name", ""))
            if not name:
                continue
            args: Any = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            calls.append(ToolCall(name, args, "llm"))
        trace = {
            "messages": messages,
            "completion": json.dumps(raw_calls, ensure_ascii=False),
            "latency_ms": latency_ms,
        }
        return self._dedupe(calls)[: self.max_calls], trace

    def _tool_messages(self, question: str) -> list[dict]:
        lines: list[str] = []
        for table in self.tables.find(question, limit=3):
            preview = "\n".join(
                " | ".join(str(c) for c in row) for row in table.rows[:3]
            )
            lines.append(
                f"- table_id={table.table_id}（{table.source} 第{table.page}页）\n"
                f"  列：{'、'.join(table.columns)}\n  示例：\n{preview}"
            )
        table_block = "\n".join(lines) or "（当前没有可用表格）"
        system = (
            "你是工具调用规划器。仅当问题确实需要精确聚合/比较/算术/单位换算时调用工具；"
            "查表 SQL 必须只读（SELECT/WITH），表名必须使用给定的 table_id。"
            "无需工具时不要发起调用。"
        )
        user = f"【可用表格】\n{table_block}\n\n【问题】\n{question}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _dedupe(calls: list[ToolCall]) -> list[ToolCall]:
        seen: set[str] = set()
        out: list[ToolCall] = []
        for call in calls:
            key = call.name + json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            out.append(call)
        return out


__all__ = [
    "Tool",
    "ToolCall",
    "ToolPlanner",
    "ToolRegistry",
    "ToolResult",
    "build_table_sql",
    "build_tool_registry",
    "convert_units",
    "evaluate_expression",
    "extract_conversion",
    "extract_expression",
    "run_python",
]
