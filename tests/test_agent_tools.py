"""B3 工具层：表格工件、DuckDB 只读查询、计算器、单位换算、代码执行与端到端工具调用。"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from core.agentic.tables import (
    DuckDBTool,
    TableRegistry,
    parse_pipe_table,
    validate_readonly_sql,
)
from core.agentic.tools import (
    ToolPlanner,
    build_table_sql,
    build_tool_registry,
    extract_conversion,
    extract_expression,
    run_python,
)
from core.pipeline import RagPipeline

HAS_DUCKDB = importlib.util.find_spec("duckdb") is not None

TABLE_TEXT = (
    "销售数据如下：\n"
    "| 产品 | 销售额 | 数量 |\n"
    "| --- | --- | --- |\n"
    "| A | 100 | 2 |\n"
    "| B | 250 | 5 |"
)

ROWS = [
    {
        "id": 10,
        "text": TABLE_TEXT,
        "source": "sales.md",
        "page": 1,
        "chunk_idx": 0,
    }
]


# ==================== 表格解析与注册表 ====================
def test_parse_pipe_table():
    parsed = parse_pipe_table(TABLE_TEXT)
    assert parsed is not None
    assert parsed["columns"] == ["产品", "销售额", "数量"]
    assert parsed["rows"] == [["A", "100", "2"], ["B", "250", "5"]]


def test_parse_pipe_table_rejects_incomplete():
    assert parse_pipe_table("只有一行 | 数据") is None
    assert parse_pipe_table("没有表格") is None
    # 列数不一致时在第一个不完整行处截断，只有表头则不成表
    assert parse_pipe_table("| a | b |\n| 1 |") is None


def test_registry_build_find_and_persist(tmp_path: Path):
    registry = TableRegistry.build(ROWS, tmp_path)
    assert len(registry) == 1
    table = registry.get("t10")
    assert table is not None
    assert table.columns == ["产品", "销售额", "数量"]
    assert table.numeric_columns == ["销售额", "数量"]
    assert Path(table.csv_path).is_file()
    assert (tmp_path / "registry.json").is_file()

    loaded = TableRegistry.load(tmp_path)
    assert loaded.get("t10") is not None
    assert loaded.get("t10").rows == table.rows

    found = registry.find("销售额总和是多少")
    assert found and found[0].table_id == "t10"


def test_registry_load_skips_corrupt_entries(tmp_path: Path):
    TableRegistry.build(ROWS, tmp_path)
    path = tmp_path / "registry.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tables"].append({"columns": ["缺少 table_id"]})
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    loaded = TableRegistry.load(tmp_path)
    assert len(loaded) == 1
    assert loaded.get("t10") is not None


def test_build_table_sql_from_question():
    table = TableRegistry.build(ROWS).get("t10")
    assert 'SUM("销售额")' in build_table_sql("A 和 B 的销售额总和是多少", table)
    assert 'MAX("数量")' in build_table_sql("数量最大是多少", table)
    assert "COUNT(*)" in build_table_sql("表格有多少行", table)
    assert "SELECT *" in build_table_sql("列出所有产品", table)


# ==================== DuckDB 只读查询 ====================
@pytest.fixture()
def registry() -> TableRegistry:
    return TableRegistry.build(ROWS)


@pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb 未安装（可选依赖）")
def test_duckdb_aggregation(registry: TableRegistry):
    tool = DuckDBTool(registry, max_rows=200, timeout_ms=3000)
    res = tool.run(
        'SELECT SUM("销售额") AS total, AVG("数量") AS avg_qty FROM "t10"',
        table_id="t10",
    )
    assert res["ok"], res
    assert res["rows"][0][0] == 350
    assert res["rows"][0][1] == 3.5

    res = tool.run(
        'SELECT "产品", "销售额" FROM "t10" ORDER BY "销售额" DESC', table_id="t10"
    )
    assert res["ok"]
    assert res["rows"][0] == ["B", 250]


@pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb 未安装（可选依赖）")
def test_duckdb_connection_reused(registry: TableRegistry, monkeypatch):
    import duckdb

    calls = {"n": 0}
    real_connect = duckdb.connect

    def counting_connect(*args, **kwargs):
        calls["n"] += 1
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", counting_connect)
    tool = DuckDBTool(registry, max_rows=200, timeout_ms=3000)

    assert tool.run('SELECT COUNT(*) FROM "t10"', table_id="t10")["ok"]
    assert tool.run('SELECT COUNT(*) FROM "t10"', table_id="t10")["ok"]
    assert calls["n"] == 1, "同一注册表应复用已加载的 DuckDB 连接"


@pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb 未安装（可选依赖）")
def test_duckdb_row_limit(registry: TableRegistry):
    tool = DuckDBTool(registry, max_rows=1, timeout_ms=3000)
    res = tool.run('SELECT * FROM "t10"', table_id="t10")
    assert res["ok"]
    assert res["row_count"] == 1
    assert res["truncated"] is True


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE t10",
        "DELETE FROM t10",
        "SELECT * FROM t10; DROP TABLE t10",
        "ATTACH 'x.db' AS x",
        "PRAGMA database_list",
        "SELECT * FROM read_csv_auto('x.csv')",
        "SELECT * FROM unknown_table",
        "CREATE TABLE x AS SELECT * FROM t10",
        "COPY (SELECT * FROM t10) TO 'x.csv'",
    ],
)
def test_dangerous_sql_rejected(registry: TableRegistry, sql: str):
    res = DuckDBTool(registry).run(sql, table_id="t10")
    assert res["ok"] is False
    assert res["error"] in {
        "unsafe_sql",
        "multi_statement",
        "table_not_allowed",
        "only_select_allowed",
    }


def test_validate_allows_trailing_semicolon_and_cte():
    allowed = {"t10"}
    assert validate_readonly_sql('SELECT COUNT(*) FROM "t10";', allowed)
    assert validate_readonly_sql(
        'WITH x AS (SELECT * FROM "t10") SELECT COUNT(*) FROM x', allowed
    )


def test_missing_duckdb_degrades(registry: TableRegistry, monkeypatch):
    monkeypatch.setitem(sys.modules, "duckdb", None)
    res = DuckDBTool(registry).run('SELECT COUNT(*) FROM "t10"', table_id="t10")
    assert res["ok"] is False
    assert res["error"] == "tool_unavailable"


def test_unknown_table_rejected_before_query(registry: TableRegistry):
    res = DuckDBTool(registry).run('SELECT * FROM "t99"', table_id="t99")
    assert res["ok"] is False
    assert res["error"] == "table_not_allowed"


# ==================== 计算器 / 单位换算 / 代码执行 ====================
def _registry(**tool_cfg) -> object:
    cfg = {
        "duckdb": {"enabled": True},
        "calculator": {"enabled": True},
        "unit_convert": {"enabled": True},
        "code_exec": {"enabled": False, "timeout_ms": 3000},
    }
    cfg.update(tool_cfg)
    return build_tool_registry(cfg, TableRegistry.build(ROWS))


def test_calculator():
    reg = _registry()
    res = reg.execute("calculator", {"expression": "(12.5 * 3 + 7) / 2"})
    assert res.ok, res.error
    assert "22.25" in res.output
    assert res.data["value"] == 22.25

    for bad in [
        "__import__('os').system('echo x')",
        "(1).__class__",
        "open('x')",
        "2**10000",
        "1/0",
    ]:
        res = reg.execute("calculator", {"expression": bad})
        assert res.ok is False, bad


def test_extract_expression_and_conversion():
    assert extract_expression("请计算 12.5 * 3 + 7 的结果") == "12.5 * 3 + 7"
    assert extract_expression("没有算术") == ""
    conv = extract_conversion("100 公里等于多少英里")
    assert conv == {"value": 100.0, "from_unit": "公里", "to_unit": "英里"}


def test_unit_convert():
    reg = _registry()
    res = reg.execute(
        "unit_convert", {"value": 100, "from_unit": "公里", "to_unit": "英里"}
    )
    assert res.ok, res.error
    assert res.data["value"] == pytest.approx(62.137119, rel=1e-5)

    res = reg.execute(
        "unit_convert", {"value": 212, "from_unit": "华氏度", "to_unit": "摄氏度"}
    )
    assert res.ok and res.data["value"] == pytest.approx(100.0)

    for args in [
        {"value": 1, "from_unit": "光年", "to_unit": "米"},
        {"value": 1, "from_unit": "公里", "to_unit": "千克"},
    ]:
        assert reg.execute("unit_convert", args).ok is False


def test_code_exec_disabled_by_default():
    reg = _registry()
    assert "code_exec" not in reg.names()
    res = reg.execute("code_exec", {"code": "print(1)"})
    assert res.ok is False and res.error == "tool_disabled"


def test_code_exec_runs_when_enabled():
    reg = _registry(code_exec={"enabled": True, "timeout_ms": 3000})
    res = reg.execute("code_exec", {"code": "print(6 * 7)"})
    assert res.ok, res.error
    assert res.output.strip() == "42"

    slow = _registry(code_exec={"enabled": True, "timeout_ms": 300})
    res = slow.execute("code_exec", {"code": "import time; time.sleep(2)"})
    assert res.ok is False and res.error == "timeout"


def test_run_python_timeout_directly():
    res = run_python("import time; time.sleep(2)", timeout_ms=200)
    assert res["ok"] is False and res["error"] == "timeout"


# ==================== 工具规划 ====================
def test_planner_rules_table_and_calculation():
    tables = TableRegistry.build(ROWS)
    reg = _registry()
    planner = ToolPlanner(reg, tables, brain=None)

    calls, trace = planner.plan("表格中 A 和 B 的销售额总和是多少", needs_table=True)
    assert trace is None
    assert calls[0].name == "duckdb_query"
    assert 'SUM("销售额")' in calls[0].arguments["sql"]
    assert calls[0].arguments["table_id"] == "t10"

    calls, _ = planner.plan("请计算 12.5 * 3 + 7", needs_calculation=True)
    assert calls[0].name == "calculator"
    assert calls[0].arguments["expression"] == "12.5 * 3 + 7"

    calls, _ = planner.plan("100 公里等于多少英里", needs_calculation=True)
    assert calls[0].name == "unit_convert"


def test_planner_uses_function_calling_when_supported():
    class ToolBrain:
        def chat_with_tools(self, messages, tools):
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": json.dumps({"expression": "6*7"}),
                        },
                    }
                ],
                "usage": {},
            }

    tables = TableRegistry.build(ROWS)
    planner = ToolPlanner(_registry(), tables, brain=ToolBrain())
    calls, trace = planner.plan("算一下 6*7", needs_calculation=True)
    assert trace is not None and trace["completion"]
    assert calls[0].name == "calculator"
    assert calls[0].arguments == {"expression": "6*7"}
    assert calls[0].reason == "llm"


# ==================== 端到端：runtime 工具节点 ====================
class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class TableStore:
    def __init__(self):
        self.rows = [dict(r) for r in ROWS] + [
            {
                "id": 11,
                "text": "补充说明：A 产品单价 50 元。",
                "source": "sales.md",
                "page": 2,
                "chunk_idx": 1,
            }
        ]

    def all_chunks(self, max_rows: int = 100000, wait_visible: bool = False) -> list[dict]:
        return [dict(r) for r in self.rows]

    def search_dense(
        self, query_vec: list[float], top_k: int = 20, filter_expr: str = ""
    ) -> list[dict]:
        hits = []
        for i, row in enumerate(self.rows):
            hit = dict(row)
            hit["dense_score"] = round(1.0 - i * 0.1, 6)
            hits.append(hit)
        return hits[:top_k]


class FakeReranker:
    enabled = False


class AnswerClient:
    def __init__(self):
        self.messages = None

    def chat(self, messages):
        self.messages = messages
        return "根据工具结果，销售额总和为 350。"


class ToolBrain:
    def chat(self, messages):
        system = messages[0]["content"]
        question = messages[-1]["content"]
        if "查询规划器" in system:
            return json.dumps(
                {
                    "intent": "fact",
                    "sub_queries": [{"text": question, "purpose": "主查询"}],
                    "needs_table": True,
                },
                ensure_ascii=False,
            )
        if "充分性评估器" in system:
            return json.dumps(
                {
                    "support": [{"sub_query": "q", "score": 0.2}] * 3,
                    "missing_aspects": ["需要精确求和"],
                },
                ensure_ascii=False,
            )
        return "{}"


def _runtime(tmp_path: Path):
    cfg = {
        "retrieval": {
            "dense_top_k": 5,
            "bm25_top_k": 5,
            "rrf_k": 60,
            "final_top_k": 3,
            "rerank": {"enabled": False},
        },
        "agent": {
            "enabled": True,
            "brain_llm": "cloud",
            "max_rounds": 3,
            "max_subqueries_per_round": 3,
            "top_k_per_subquery": 3,
            "max_final_evidence": 6,
            "on_budget_exhausted": "partial",
            "analyze": {"mode": "brain", "max_retries": 0},
            "routing": {"keyword_bm25_first": True, "summary_evidence_bonus": 2},
            "sufficiency": {
                "tau_low": 0.45,
                "tau_high": 0.65,
                "weights": {
                    "claim_support": 0.45,
                    "coverage": 0.25,
                    "top_score": 0.20,
                    "diversity": 0.10,
                },
            },
            "hyde": {"enabled": False},
            "budget": {
                "max_llm_calls": 8,
                "max_brain_calls": 8,
                "max_wall_ms": 30000,
                "max_tool_calls": 3,
                "novelty_epsilon": 0.10,
            },
            "tools": {
                "enabled": True,
                "tables_dir": str(tmp_path / "tables"),
                "duckdb": {"enabled": True, "max_rows": 200, "timeout_ms": 3000},
                "calculator": {"enabled": True},
                "unit_convert": {"enabled": True},
                "code_exec": {"enabled": False},
            },
            "trace": {"enabled": True, "dir": str(tmp_path / "traces")},
        },
    }
    answer = AnswerClient()
    pipeline = RagPipeline(
        cfg,
        store=TableStore(),
        embedder=FakeEmbedder(),
        llm=answer,
        reranker=FakeReranker(),
    )
    llms = {"cloud": {"client": ToolBrain()}, "ft": {"client": answer}}
    from core.agentic.runtime import AgentRuntime

    return AgentRuntime(cfg, pipeline, llms), answer


def _events(out, stage):
    return [e for e in out["trace"] if e["stage"] == stage]


@pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb 未安装（可选依赖）")
def test_runtime_tool_node_aggregates(tmp_path: Path):
    runtime, answer = _runtime(tmp_path)
    out = runtime.ask("表格中 A 和 B 的销售额总和是多少", session_id="b3a", llm="ft", include_trace=True)

    assert out["status"] == "partial"
    assert out["metrics"]["tool_calls"] == 1
    result = out["tool_results"][0]
    assert result["tool"] == "duckdb_query"
    assert result["ok"] is True
    assert "350" in result["output"]

    # 工具结果注入 Compose，且 trace 记录工具事件
    assert "【工具结果" in answer.messages[-1]["content"]
    assert "350" in answer.messages[-1]["content"]
    assert len(_events(out, "tool")) == 1
    assert _events(out, "table_extract")


def test_runtime_tool_node_degrades_without_duckdb(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "duckdb", None)
    runtime, answer = _runtime(tmp_path)
    out = runtime.ask("表格中 A 和 B 的销售额总和是多少", session_id="b3b", llm="ft", include_trace=True)

    assert out["answer"]
    assert out["tool_results"][0]["ok"] is False
    assert out["tool_results"][0]["error"] == "tool_unavailable"
    # 失败工具结果不注入 Compose
    assert "【工具结果" not in answer.messages[-1]["content"]
