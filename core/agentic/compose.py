"""Compose 层：B0 提供直答与基础带证据回答；B4 扩展逐句引用与冲突并列。"""
from __future__ import annotations

from core.llm import build_messages

DIRECT_SYSTEM_PROMPT = """你是文档问答助手。当前问题属于闲聊或系统能力询问，不需要检索文档。
直接、简洁地回答；不要声称“资料不足”，也不要编造具体的文档内容。"""

CITED_SYSTEM_PROMPT = """你是一个严谨的中文文档问答助手。请仅依据【参考资料】回答用户问题。
每个包含事实的句子末尾必须标注有效引用编号，如 [E1]；同一句可标多个，如 [E1][E2]。
工具结果使用 [T1] 标注；不要引用不存在的编号。
若资料不足以回答，请直接说明“根据现有资料无法回答”，不要编造。
若存在【冲突提示】，必须并列展示不同说法及各自来源，不得强行合并。"""


def direct_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def ok_tool_results(tool_results: list[dict] | None) -> list[dict]:
    """本轮可展示给模型的有效工具结果（verify 的工具编号必须复用同一规则）。"""
    return [r for r in (tool_results or []) if r.get("ok") and r.get("output")]


def _conflict_block(conflicts: list[dict] | None) -> str:
    items = [c for c in (conflicts or []) if c.get("claim")]
    if not items:
        return ""
    lines = []
    for item in items[:3]:
        ids = [str(x) for x in item.get("evidence_ids", []) if x]
        suffix = f"（涉及 {'、'.join(ids)}）" if ids else ""
        lines.append(f"- {item['claim']}{suffix}")
    return (
        "【冲突提示】\n"
        + "\n".join(lines)
        + "\n请在答案中并列展示冲突各方的说法及来源，不要强行合并。"
    )


def cited_messages(
    question: str,
    hits: list[dict],
    tool_results: list[dict] | None = None,
    conflicts: list[dict] | None = None,
) -> list[dict]:
    """逐句引用提示：证据块使用 E 编号，工具结果使用 T 编号。"""
    context_lines = []
    for i, hit in enumerate(hits, 1):
        eid = hit.get("evidence_id") or f"E{i}"
        context_lines.append(
            f"[{eid}] （来源: {hit.get('source')} 第{hit.get('page')}页）\n"
            f"{hit.get('text', '')}"
        )
    sections = ["【参考资料】\n" + ("\n\n".join(context_lines) or "（无检索结果）")]

    tools = ok_tool_results(tool_results)
    if tools:
        tool_lines = [
            f"[T{i}] {r.get('tool', 'tool')}：{r.get('output', '')}"
            for i, r in enumerate(tools, 1)
        ]
        sections.append("【工具结果（确定性计算）】\n" + "\n".join(tool_lines))

    conflict_text = _conflict_block(conflicts)
    if conflict_text:
        sections.append(conflict_text)

    user = "\n\n".join(sections) + f"\n\n【问题】\n{question}"
    return [
        {"role": "system", "content": CITED_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def grounded_messages(
    question: str,
    hits: list[dict],
    tool_results: list[dict] | None = None,
    *,
    cite_evidence: bool = False,
    conflicts: list[dict] | None = None,
) -> list[dict]:
    """复用现有 RAG 提示词，保证 B0 与固定流水线的生成口径一致。

    工具结果作为【工具结果】附加在用户消息末尾；失败结果不注入，避免污染答案。
    ``cite_evidence=True`` 时切换到逐句引用提示（B4），旧接口提示词不变。
    """
    if cite_evidence:
        return cited_messages(question, hits, tool_results, conflicts)

    messages = build_messages(question, hits)
    ok_results = ok_tool_results(tool_results)
    if not ok_results:
        return messages
    lines = [
        f"[工具 {i}] {r.get('tool', 'tool')}：{r.get('output', '')}"
        for i, r in enumerate(ok_results, 1)
    ]
    block = (
        "\n\n【工具结果（确定性计算，可作为事实依据）】\n"
        + "\n".join(lines)
        + "\n请结合工具结果回答；工具结果是精确计算值，优先于文本中的估算。"
    )
    messages[-1] = {
        "role": messages[-1]["role"],
        "content": messages[-1]["content"] + block,
    }
    return messages


def refusal_text(reason: str = "") -> str:
    text = "根据当前知识库的检索结果，无法可靠回答这个问题。"
    if reason:
        text += f"（{reason}）"
    return text
