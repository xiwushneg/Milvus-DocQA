"""表格解析与注册表：管道表格、数值列判定、持久化与损坏容错。"""
from __future__ import annotations

import json

import pytest

from core.agentic.tables import (
    DuckDBTool,
    TableArtifact,
    TableRegistry,
    parse_pipe_table,
    to_number,
    validate_readonly_sql,
)

TABLE_TEXT = """说明文字

| 城市 | GDP（万亿） | 人口（万） |
| --- | --- | --- |
| 上海 | 4.7 | 2487 |
| 北京 | 4.4 | 2189 |
"""


def test_parse_pipe_table_returns_columns_and_rows():
    parsed = parse_pipe_table(TABLE_TEXT)
    assert parsed is not None
    assert parsed["columns"] == ["城市", "GDP（万亿）", "人口（万）"]
    assert len(parsed["rows"]) == 2
    assert parsed["rows"][0] == ["上海", "4.7", "2487"]


def test_parse_pipe_table_rejects_incomplete_tables():
    assert parse_pipe_table("") is None
    assert parse_pipe_table("没有表格") is None
    # 只有表头（无数据行）不成表
    assert parse_pipe_table("| a | b |\n| --- | --- |") is None
    # 列数不一致时在第一个不完整行处截断
    ragged = "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 |"
    assert parse_pipe_table(ragged)["rows"] == [["1", "2"]]


def test_to_number_supports_thousands_and_percent():
    assert to_number("1,234") == 1234.0
    assert to_number("12.5%") == 0.125
    assert to_number("--") is None
    assert to_number("N/A") is None


def test_artifact_numeric_columns_and_types():
    artifact = TableArtifact(
        table_id="t1",
        columns=["城市", "GDP", "备注"],
        rows=[["上海", "4.7", ""], ["北京", "4.4", ""]],
    )
    assert artifact.numeric_columns == ["GDP"]
    assert artifact.column_types() == ["VARCHAR", "DOUBLE", "VARCHAR"]


def test_registry_save_and_load_roundtrip(tmp_path):
    artifact = TableArtifact(
        table_id="t1",
        source="doc_a.md",
        page=2,
        chunk_idx=3,
        columns=["城市", "GDP"],
        rows=[["上海", "4.7"], ["北京", "4.4"]],
    )
    registry = TableRegistry([artifact])
    registry.save(tmp_path)

    loaded = TableRegistry.load(tmp_path)
    assert len(loaded) == 1
    restored = loaded.get("t1")
    assert restored is not None
    assert restored.columns == ["城市", "GDP"]
    assert restored.rows == [["上海", "4.7"], ["北京", "4.4"]]
    assert (tmp_path / "registry.json").is_file()


def test_registry_load_tolerates_corrupt_entries(tmp_path):
    payload = {
        "version": 1,
        "tables": [
            {"table_id": "ok", "columns": ["a"], "rows": [["1"]]},
            {"columns": ["missing_id"]},
        ],
    }
    (tmp_path / "registry.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    registry = TableRegistry.load(tmp_path)
    assert [t.table_id for t in registry.all()] == ["ok"]

    (tmp_path / "registry.json").write_text("{broken", encoding="utf-8")
    assert TableRegistry.load(tmp_path).all() == []


def test_validate_sql_allows_keywords_inside_string_literals():
    sql = "SELECT 'set load copy from t1; -- x' AS label FROM t1"
    assert validate_readonly_sql(sql, {"t1"}) == sql
    # 字面量里的 '' 转义不影响判定
    assert validate_readonly_sql("SELECT 'it''s set' AS v FROM t1", {"t1"}).endswith("t1")


def test_validate_sql_still_rejects_unsafe_outside_literals():
    with pytest.raises(ValueError, match="unsafe_sql"):
        validate_readonly_sql("COPY t1 TO 'x.csv'", {"t1"})
    with pytest.raises(ValueError, match="multi_statement"):
        validate_readonly_sql("SELECT 1; SELECT 2", {"t1"})
    with pytest.raises(ValueError, match="comments_not_allowed"):
        validate_readonly_sql("SELECT 1 -- x", {"t1"})
    with pytest.raises(ValueError, match="table_not_allowed"):
        validate_readonly_sql("SELECT 'from t9' FROM t9", {"t1"})
    with pytest.raises(ValueError, match="only_select_allowed"):
        validate_readonly_sql("VALUES (1)", {"t1"})


def test_duckdb_run_accepts_literal_with_forbidden_words():
    duckdb = pytest.importorskip("duckdb")
    assert duckdb
    artifact = TableArtifact(
        table_id="t1", columns=["a"], rows=[["set"], ["load"]]
    )
    tool = DuckDBTool(TableRegistry([artifact]))
    out = tool.run("SELECT 'set load copy' AS label FROM t1 LIMIT 1")
    assert out["ok"] is True, out["error"]
    assert out["rows"][0][0] == "set load copy"
