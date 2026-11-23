"""B0 状态机单测：线性、条件环、最大步数、节点异常。"""
from __future__ import annotations

from core.agentic.graph import END, AgentGraph
from core.agentic.state import AgentState, Status
from core.agentic.trace import TraceRecorder


def _state() -> AgentState:
    return AgentState.new("测试问题")


def test_linear_flow_and_trace():
    order: list[str] = []

    def first(s: AgentState) -> AgentState:
        order.append("first")
        return s

    def second(s: AgentState) -> AgentState:
        order.append("second")
        s.answer = "ok"
        return s

    graph = AgentGraph(
        {"first": first, "second": second},
        {"first": lambda s: "second", "second": lambda s: END},
        entry="first",
    )
    recorder = TraceRecorder()
    out = graph.run(_state(), recorder)

    assert order == ["first", "second"]
    assert out.answer == "ok"
    assert [e["stage"] for e in recorder.events].count("node") == 2


def test_conditional_loop_is_real():
    rounds: list[int] = []

    def retrieve(s: AgentState) -> AgentState:
        s.round += 1
        rounds.append(s.round)
        return s

    def assess(s: AgentState) -> AgentState:
        return s

    def compose(s: AgentState) -> AgentState:
        s.answer = "done"
        return s

    graph = AgentGraph(
        {"retrieve": retrieve, "assess": assess, "compose": compose},
        {
            "retrieve": lambda s: "assess",
            "assess": lambda s: "retrieve" if s.round < 3 else "compose",
            "compose": lambda s: END,
        },
        entry="retrieve",
    )
    out = graph.run(_state())

    assert rounds == [1, 2, 3]
    assert out.answer == "done"


def test_max_steps_guard():
    def loop(s: AgentState) -> AgentState:
        s.round += 1
        return s

    graph = AgentGraph({"loop": loop}, {"loop": lambda s: "loop"}, entry="loop", max_steps=4)
    recorder = TraceRecorder()
    out = graph.run(_state(), recorder)

    assert out.status == Status.ERROR
    assert out.stop_reason == "max_steps"
    assert any(e["stage"] == "graph_stop" for e in recorder.events)


def test_node_exception_is_captured():
    def boom(s: AgentState) -> AgentState:
        raise RuntimeError("模拟节点失败")

    graph = AgentGraph({"boom": boom}, {"boom": lambda s: END}, entry="boom")
    recorder = TraceRecorder()
    out = graph.run(_state(), recorder)

    assert out.status == Status.ERROR
    assert out.stop_reason == "node_error"
    assert "模拟节点失败" in out.error
    assert any(e["stage"] == "node_error" for e in recorder.events)


def test_node_retry_recovers_transient_error():
    calls = {"n": 0}

    def flaky(s: AgentState) -> AgentState:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("瞬时失败")
        s.answer = "ok"
        return s

    graph = AgentGraph(
        {"flaky": flaky},
        {"flaky": lambda s: END},
        entry="flaky",
        node_retries=1,
    )
    recorder = TraceRecorder()
    out = graph.run(_state(), recorder)

    assert out.status != Status.ERROR
    assert out.answer == "ok"
    assert calls["n"] == 2
    assert [e["stage"] for e in recorder.events].count("node_retry") == 1


def test_node_retry_exhausted_reports_error():
    def boom(s: AgentState) -> AgentState:
        raise RuntimeError("持续失败")

    graph = AgentGraph(
        {"boom": boom}, {"boom": lambda s: END}, entry="boom", node_retries=2
    )
    out = graph.run(_state())

    assert out.status == Status.ERROR
    assert out.stop_reason == "node_error"


def test_unknown_node_is_rejected():
    graph = AgentGraph({"start": lambda s: s}, {"start": lambda s: "missing"}, entry="start")
    out = graph.run(_state())
    assert out.status == Status.ERROR
    assert out.stop_reason == "unknown_node"
