"""AgentState 基础对象：预算、用量汇总与过滤表达式。"""
from __future__ import annotations

import time

import pytest

from core.agentic.state import Budget, Usage, build_filter_expr


def test_budget_from_config_maps_all_limits():
    budget = Budget.from_config(
        {
            "max_rounds": 2,
            "budget": {
                "max_llm_calls": 5,
                "max_brain_calls": 4,
                "max_prompt_tokens": 900,
                "max_completion_tokens": 300,
                "max_wall_ms": 1500,
                "max_tool_calls": 2,
                "novelty_epsilon": 0.2,
            },
        }
    )
    assert budget == Budget(
        max_rounds=2,
        max_llm_calls=5,
        max_brain_calls=4,
        max_prompt_tokens=900,
        max_completion_tokens=300,
        max_wall_ms=1500,
        max_tool_calls=2,
        novelty_epsilon=0.2,
    )


def test_usage_sums_brain_and_answer():
    usage = Usage()
    usage.brain.add(10, 2, 5, estimated=False)
    usage.answer.add(20, 3, 7, estimated=True)
    usage.tool_calls = 1

    assert usage.llm_calls == 2
    assert usage.prompt_tokens == 30
    assert usage.completion_tokens == 5
    assert usage.to_dict()["tool_calls"] == 1


def test_budget_can_continue_stops_on_each_limit():
    started = time.perf_counter()
    usage = Usage()
    budget = Budget(
        max_llm_calls=2,
        max_brain_calls=1,
        max_prompt_tokens=100,
        max_completion_tokens=100,
        max_tool_calls=1,
        max_wall_ms=10000,
    )
    assert budget.can_continue(usage, started)

    usage.brain.calls = 1
    assert not budget.can_continue(usage, started), "brain 调用达到上限应停止"

    usage.brain.calls = 0
    usage.answer.calls = 2
    assert not budget.can_continue(usage, started), "总 LLM 调用达到上限应停止"

    usage.answer.calls = 0
    usage.brain.prompt_tokens = 100
    assert not budget.can_continue(usage, started), "prompt token 达到上限应停止"


def test_budget_can_continue_stops_on_wall_clock():
    budget = Budget(max_wall_ms=1)
    time.sleep(0.01)
    assert not budget.can_continue(Usage(), time.perf_counter() - 0.05)


def test_build_filter_expr_escapes_and_validates():
    assert build_filter_expr('a\\b"c', page=2) == 'source == "a\\\\b\\"c" and page == 2'
    assert build_filter_expr() == ""
    with pytest.raises(ValueError):
        build_filter_expr("")
    with pytest.raises(ValueError):
        build_filter_expr(page=-1)
    with pytest.raises(ValueError):
        build_filter_expr(page=True)  # bool 是 int 子类，必须拒绝
