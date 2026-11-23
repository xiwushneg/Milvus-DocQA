"""DOCX 解析：可选按正文顺序保留段落/表格。"""
from __future__ import annotations

import io

import pytest

from core.loader import parse_bytes

docx = pytest.importorskip("docx")


def _docx_bytes() -> bytes:
    document = docx.Document()
    document.add_paragraph("开头段落")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "指标"
    table.rows[0].cells[1].text = "数值"
    document.add_paragraph("结尾段落")
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_preserve_docx_order_keeps_body_sequence():
    text = parse_bytes(
        "a.docx", _docx_bytes(), preserve_docx_order=True
    )[0]["text"]
    assert text.index("开头段落") < text.index("指标 | 数值") < text.index("结尾段落")


def test_default_docx_order_appends_tables_at_end():
    text = parse_bytes("a.docx", _docx_bytes())[0]["text"]
    assert text.index("结尾段落") < text.index("指标 | 数值")
