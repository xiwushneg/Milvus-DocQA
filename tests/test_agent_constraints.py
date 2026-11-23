"""硬约束校验：过滤字段白名单、无 schema 变更、无编排框架依赖。"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.agentic.state import SubQuery, build_filter_expr

ROOT = Path(__file__).resolve().parents[1]


def test_filter_expr_only_supports_source_and_page():
    assert build_filter_expr("demo.md", 2) == 'source == "demo.md" and page == 2'
    assert build_filter_expr(source="demo.md") == 'source == "demo.md"'
    assert build_filter_expr(page=0) == "page == 0"
    assert build_filter_expr() == ""


def test_filter_expr_escapes_source():
    expr = build_filter_expr('a"b\\c')
    assert expr == 'source == "a\\"b\\\\c"'


@pytest.mark.parametrize("page", [-1, 1.5, True])
def test_filter_expr_rejects_invalid_page(page):
    with pytest.raises(ValueError):
        build_filter_expr(page=page)


def test_filter_expr_rejects_empty_source():
    with pytest.raises(ValueError):
        build_filter_expr(source="   ")


def test_subquery_filter_expr():
    sq = SubQuery(text="第 2 页", source="demo.md", page=2)
    assert sq.filter_expr == 'source == "demo.md" and page == 2'


def test_no_orchestration_framework_dependency():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "langgraph" not in requirements
    assert "langchain" not in requirements
    for path in (ROOT / "core" / "agentic").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith(("langgraph", "langchain"))
                    for alias in node.names
                ), f"{path.name} 导入了编排框架"
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    ("langgraph", "langchain")
                ), f"{path.name} 导入了编排框架"


def test_agentic_layer_never_mutates_milvus_schema():
    forbidden = ("create_collection", "create_index", "add_field", "drop_collection")
    for path in (ROOT / "core" / "agentic").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} 出现 Milvus schema 变更: {token}"
