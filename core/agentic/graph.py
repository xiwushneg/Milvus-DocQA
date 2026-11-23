"""极简显式状态机：节点注册 + 条件边 + 最大步数守卫。

节点签名固定为 AgentState -> AgentState，依赖通过闭包注入；
这样节点可单测、图可回放，也避免引入 LangGraph 等编排依赖。
"""
from __future__ import annotations

import time
from typing import Callable

from core.agentic.state import AgentState, Status
from core.agentic.trace import TraceRecorder

END = "__end__"
Node = Callable[[AgentState], AgentState]
Edge = Callable[[AgentState], str]


class AgentGraph:
    def __init__(
        self,
        nodes: dict[str, Node],
        edges: dict[str, Edge],
        *,
        entry: str = "normalize",
        max_steps: int = 50,
        node_retries: int = 0,
    ):
        if entry not in nodes:
            raise ValueError(f"入口节点不存在: {entry}")
        self.nodes = nodes
        self.edges = edges
        self.entry = entry
        self.max_steps = max_steps
        self.node_retries = max(0, int(node_retries))

    def run(self, state: AgentState, recorder: TraceRecorder | None = None) -> AgentState:
        current = self.entry
        steps = 0

        while current != END:
            if steps >= self.max_steps:
                state.status = Status.ERROR
                state.stop_reason = "max_steps"
                state.error = f"状态机超过最大步数 {self.max_steps}"
                if recorder:
                    recorder.emit(
                        "graph_stop",
                        status="error",
                        payload={"reason": "max_steps", "steps": steps},
                    )
                break

            node = self.nodes.get(current)
            if node is None:
                state.status = Status.ERROR
                state.stop_reason = "unknown_node"
                state.error = f"未知节点: {current}"
                if recorder:
                    recorder.emit(
                        "graph_stop",
                        status="error",
                        payload={"reason": "unknown_node", "node": current},
                    )
                break

            t0 = time.perf_counter()
            error: Exception | None = None
            for attempt in range(self.node_retries + 1):
                try:
                    state = node(state)
                    error = None
                    break
                except Exception as exc:
                    error = exc
                    if attempt < self.node_retries and recorder:
                        recorder.emit(
                            "node_retry",
                            status="retry",
                            round=state.round,
                            latency_ms=int((time.perf_counter() - t0) * 1000),
                            payload={
                                "node": current,
                                "attempt": attempt + 1,
                                "error": str(exc),
                            },
                        )
            if error is not None:  # 重试耗尽：节点失败不拖垮服务，记录后终止
                state.status = Status.ERROR
                state.stop_reason = "node_error"
                state.error = f"{current}: {error}"
                if recorder:
                    recorder.emit(
                        "node_error",
                        status="error",
                        round=state.round,
                        latency_ms=int((time.perf_counter() - t0) * 1000),
                        payload={"node": current, "error": str(error)},
                    )
                break

            latency_ms = int((time.perf_counter() - t0) * 1000)
            if recorder:
                recorder.emit(
                    "node",
                    round=state.round,
                    latency_ms=latency_ms,
                    payload={"node": current},
                )

            edge = self.edges.get(current)
            nxt = edge(state) if edge else END
            current = nxt or END
            steps += 1

        return state
