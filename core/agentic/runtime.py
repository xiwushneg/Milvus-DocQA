"""AgentRuntime：装配状态机（直答 / 智能路由 / 自反思检索闭环）。

B2 能力：检索 → 充分性评估 → 缺口改写 → 再检索，受轮次、LLM/耗时预算与
检索新颖度约束；证据跨轮累加且 E1..En 编号稳定。
"""
from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any

from core.agentic.analyze import HydeGenerator, build_analyzer
from core.agentic.assess import build_assessor
from core.agentic.chunk_types import ChunkTypeIndex, count_types
from core.agentic.compose import direct_messages, grounded_messages, refusal_text
from core.agentic.graph import END, AgentGraph
from core.agentic.memory import SessionStore, detect_reference
from core.agentic.retrieval import RetrievalAdapter
from core.agentic.rewrite import Rewriter
from core.agentic.route import route_subquery
from core.agentic.state import (
    AgentState,
    Budget,
    Decision,
    Evidence,
    Intent,
    Status,
)
from core.agentic.tables import DuckDBTool, TableRegistry
from core.agentic.tools import ToolPlanner, build_tool_registry
from core.agentic.trace import TraceRecorder
from core.agentic.verify import build_verifier
from core.llm import build_rewrite_messages

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Agentic RAG 运行入口。

    B1/B2：指代消解 → brain 结构化分析 → 路由 → 检索 → 充分性评估 →
    缺口改写/再检索循环 → 分角色生成。
    """

    def __init__(self, cfg: dict, pipeline, llms: dict[str, dict], analyzer=None):
        acfg = cfg.get("agent", {}) or {}
        if not acfg.get("enabled", False):
            raise ValueError("agent.enabled=false，不应构建 AgentRuntime")
        self.cfg = cfg
        self.pipeline = pipeline
        self.llms = llms
        self.acfg = acfg

        brain_id = acfg.get("brain_llm", "cloud")
        self.brain_id = brain_id if brain_id in llms else "cloud"
        self.brain = llms[self.brain_id]["client"]

        self.adapter = RetrievalAdapter.from_pipeline(pipeline)
        self.analyzer = analyzer or build_analyzer(cfg, self.brain)
        self.assessor = build_assessor(
            cfg, self.brain, rerank_mode=getattr(pipeline.reranker, "mode", "api")
        )
        self.rewriter = Rewriter(
            self.brain,
            max_subqueries=int(acfg.get("max_subqueries_per_round", 3)),
        )
        self.on_budget_exhausted = str(acfg.get("on_budget_exhausted", "partial"))
        self.chunk_types = ChunkTypeIndex(
            lambda: getattr(self.adapter.bm25, "_meta", []) or []
        )
        self.budget_template = Budget.from_config(acfg)

        self.hyde_cfg = acfg.get("hyde", {}) or {}
        self.hyde = (
            HydeGenerator(self.brain, max_chars=int(self.hyde_cfg.get("max_chars", 400)))
            if self.hyde_cfg.get("enabled", False)
            else None
        )
        self.routing_cfg = acfg.get("routing", {}) or {}
        self.summary_bonus = int(self.routing_cfg.get("summary_evidence_bonus", 2))
        self.keyword_bm25_first = bool(
            self.routing_cfg.get("keyword_bm25_first", True)
        )

        # B4：会话记忆 + 逐句引用校验（citations.enabled=true 时才启用）
        mem_cfg = acfg.get("memory", {}) or {}
        self.history_turns = int(mem_cfg.get("history_turns", 6))
        self.sessions = SessionStore(
            ttl_s=int(mem_cfg.get("session_ttl_s", 7200)),
            max_turns=int(mem_cfg.get("max_turns", 20)),
            persist_path=(mem_cfg.get("persist_path") or None),
        )
        cit_cfg = acfg.get("citations", {}) or {}
        self.citations_enabled = bool(cit_cfg.get("enabled", False))
        self.verifier = build_verifier(cfg, self.brain)

        trace_cfg = acfg.get("trace", {}) or {}
        self.trace_enabled = bool(trace_cfg.get("enabled", True))
        self.trace_dir = trace_cfg.get("dir", "data/traces")

        # 工具层：duckdb 可选依赖；表格注册表优先读预构建工件，缺失时懒加载
        tools_cfg = acfg.get("tools", {}) or {}
        self.tools_enabled = bool(tools_cfg.get("enabled", False))
        self.tables_dir = str(tools_cfg.get("tables_dir", "data/tables"))
        self.tables = TableRegistry.load(self.tables_dir)
        self.duckdb_tool = DuckDBTool(
            self.tables,
            max_rows=int((tools_cfg.get("duckdb", {}) or {}).get("max_rows", 200)),
            timeout_ms=int((tools_cfg.get("duckdb", {}) or {}).get("timeout_ms", 3000)),
        )
        self.tool_registry = build_tool_registry(
            tools_cfg, self.tables, duckdb_tool=self.duckdb_tool
        )
        self.tool_planner = ToolPlanner(
            self.tool_registry,
            self.tables,
            brain=self.brain,
            max_calls=int(self.budget_template.max_tool_calls),
        )

        cost_cfg = acfg.get("cost", {}) or {}
        self.prompt_per_1k = float(cost_cfg.get("prompt_per_1k", 0.0))
        self.completion_per_1k = float(cost_cfg.get("completion_per_1k", 0.0))

    # ==================== 对外入口 ====================
    def ask(
        self,
        question: str,
        *,
        session_id: str = "",
        llm: str = "cloud",
        top_k: int | None = None,
        history: list[dict] | None = None,
        include_trace: bool = False,
    ) -> dict[str, Any]:
        recorder = TraceRecorder(session_id=session_id, enabled=self.trace_enabled)
        result = self._execute(
            question,
            recorder=recorder,
            session_id=session_id,
            llm=llm,
            top_k=top_k,
            history=history,
            include_trace=include_trace,
        )
        result.pop("_streamed", None)
        return result

    def ask_stream(
        self,
        question: str,
        *,
        session_id: str = "",
        llm: str = "cloud",
        top_k: int | None = None,
        history: list[dict] | None = None,
    ):
        """流式入口：先 yield session，再实时推送 trace/delta，最后 refs + done。

        状态机是同步的，这里用后台线程执行，主生成器只从队列取事件；
        FastAPI 侧用 ``iterate_in_threadpool`` 消费，不阻塞事件循环。
        """
        import queue
        import threading

        chosen_id = llm if llm in self.llms else "cloud"
        recorder = TraceRecorder(session_id=session_id, enabled=self.trace_enabled)
        q: queue.Queue = queue.Queue()

        def worker() -> None:
            try:
                out = self._execute(
                    question,
                    recorder=recorder,
                    session_id=session_id,
                    llm=llm,
                    top_k=top_k,
                    history=history,
                    on_event=q.put,
                    token_cb=lambda text: q.put({"type": "delta", "text": text}),
                )
                q.put(
                    {
                        "type": "refs",
                        "references": out["references"],
                        "citations": out["citations"],
                        "conflicts": out["conflicts"],
                        "tool_results": out["tool_results"],
                        "confidence": out["confidence"],
                    }
                )
                streamed = bool(out.pop("_streamed", False))
                if not streamed:
                    q.put({"type": "delta", "text": out["answer"]})
                q.put({"type": "done", **out})
            except Exception as exc:  # 后台线程异常必须显式回传
                logger.exception("agent 流式执行失败")
                q.put(
                    {
                        "type": "error",
                        "detail": str(exc),
                        "trace_id": recorder.trace_id,
                        "llm": chosen_id,
                    }
                )
            finally:
                q.put(None)

        threading.Thread(target=worker, name="agent-stream", daemon=True).start()
        yield {
            "type": "session",
            "session_id": session_id,
            "trace_id": recorder.trace_id,
            "llm": chosen_id,
        }
        while True:
            item = q.get()
            if item is None:
                break
            yield item

    def _execute(
        self,
        question: str,
        *,
        recorder: TraceRecorder,
        session_id: str = "",
        llm: str = "cloud",
        top_k: int | None = None,
        history: list[dict] | None = None,
        include_trace: bool = False,
        on_event=None,
        token_cb=None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        chosen_id = llm if llm in self.llms else "cloud"
        answer_client = self.llms[chosen_id]["client"]
        recorder.on_event = on_event
        streamed = {"count": 0}

        state = AgentState.new(
            question,
            session_id=session_id,
            llm_id=chosen_id,
            top_k=top_k,
            history=history,
            budget=self.budget_template,
            trace_id=recorder.trace_id,
        )
        # 预算/用量统一由 recorder 累计；让 state.usage 指向同一对象，
        # 避免 _after_assess 与工具边读到永远为 0 的默认 Usage。
        state.usage = recorder.usage

        nodes = self._build_nodes(recorder, answer_client, token_cb, streamed)
        graph = AgentGraph(
            nodes,
            self._build_edges(),
            entry="normalize",
            node_retries=int(self.acfg.get("node_retries", 1)),
        )
        state = graph.run(state, recorder)

        if session_id and state.status != Status.ERROR:
            self._remember_session(state, recorder)

        if state.status == Status.ERROR:
            recorder.finalize(Status.ERROR, state.stop_reason)
        else:
            recorder.finalize(state.status, state.stop_reason)

        usage = recorder.usage
        cost = (
            usage.prompt_tokens / 1000 * self.prompt_per_1k
            + usage.completion_tokens / 1000 * self.completion_per_1k
        )
        refused = state.status == Status.REFUSED
        metrics = {
            "rounds": state.round,
            "sub_queries": len(state.plan),
            "rewrites": state.rewrite_count,
            "sufficiency_score": (
                round(state.sufficiency.score, 4) if state.sufficiency else 0.0
            ),
            "citation_coverage": round(state.citation_coverage, 4),
            "invalid_citations": len(state.invalid_citations),
            "confidence": round(state.confidence, 4),
            "tool_calls": usage.tool_calls,
            "brain_llm_calls": usage.brain.calls,
            "brain_latency_ms": usage.brain.latency_ms,
            "answer_llm_calls": usage.answer.calls,
            "answer_latency_ms": usage.answer.latency_ms,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cost_cny": round(cost, 6),
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "stop_reason": state.stop_reason,
        }

        if self.trace_enabled:
            try:
                recorder.write(self.trace_dir)
            except Exception as exc:  # 轨迹落盘失败不影响答案返回
                logger.warning("trace 写入失败: %s", exc)

        result: dict[str, Any] = {
            "answer": state.answer,
            "references": state.references,
            "citations": state.citations,
            "conflicts": state.conflicts,
            "tool_results": state.tool_results,
            "confidence": round(state.confidence, 4),
            "refused": refused,
            "refusal_reason": state.stop_reason if refused else None,
            "session_id": session_id,
            "trace_id": recorder.trace_id,
            "metrics": metrics,
            "llm": chosen_id,
            "status": state.status.value,
            "error": state.error,
            "_streamed": streamed["count"] > 0,
        }
        if include_trace:
            result["trace"] = recorder.events
        return result

    # ==================== 节点 ====================
    def _build_nodes(
        self,
        recorder: TraceRecorder,
        answer_client,
        token_cb=None,
        streamed: dict | None = None,
    ) -> dict:
        streamed = streamed if streamed is not None else {"count": 0}
        import inspect

        def _accepts(fn, name: str) -> bool:
            try:
                return name in inspect.signature(fn).parameters
            except (TypeError, ValueError):
                return False

        accepts_chat_usage = _accepts(answer_client.chat, "usage")

        def generate(messages: list[dict]) -> tuple[str, dict]:
            """生成最终答案；流式可用时逐段回调，否则退回一次性生成。

            返回 ``(answer, usage)``，usage 为空表示服务端未返回真实用量。
            """
            usage: dict = {}
            can_stream = token_cb is not None and hasattr(answer_client, "chat_stream")
            if can_stream:
                try:
                    if _accepts(answer_client.chat_stream, "usage"):
                        iterator = answer_client.chat_stream(messages, usage=usage)
                    else:
                        iterator = answer_client.chat_stream(messages)
                    parts: list[str] = []
                    for piece in iterator:
                        if not piece:
                            continue
                        parts.append(piece)
                        streamed["count"] += 1
                        token_cb(piece)
                    return "".join(parts), usage
                except Exception as exc:  # 流式中断时退回非流式，保证答案完整
                    logger.warning("流式生成失败，回退非流式: %s", exc)
                    usage.clear()

            if accepts_chat_usage:
                return answer_client.chat(messages, usage=usage), usage
            return answer_client.chat(messages), usage

        def normalize(state: AgentState) -> AgentState:
            session = (
                self.sessions.get(state.session_id) if state.session_id else None
            )
            if session is not None and not state.history:
                state.history = session.history(self.history_turns)

            reference = detect_reference(state.question)
            if reference["anaphora"] and session is not None:
                state.focus_source = session.focus_source or None
                state.focus_page = reference["page"]

            if state.history:
                messages = build_rewrite_messages(state.history, state.question)
                t0 = time.perf_counter()
                try:
                    rewritten = self.brain.chat(messages).strip().strip('"“”')
                    if rewritten and len(rewritten) <= 300:
                        state.standalone_question = rewritten
                    recorder.record_llm(
                        "brain",
                        messages,
                        state.standalone_question,
                        int((time.perf_counter() - t0) * 1000),
                    )
                except Exception as exc:
                    recorder.emit(
                        "rewrite",
                        status="error",
                        payload={"error": str(exc)},
                    )
            recorder.emit(
                "memory",
                payload={
                    "session_id": state.session_id,
                    "turns": len(session.turns) if session else 0,
                    "history_used": bool(state.history),
                    "anaphora": reference["anaphora"],
                    "focus_source": state.focus_source,
                    "focus_page": state.focus_page,
                },
            )
            recorder.emit(
                "normalize",
                payload={
                    "question": state.question,
                    "standalone_question": state.standalone_question,
                },
            )
            return state

        def analyze(state: AgentState) -> AgentState:
            analysis = self.analyzer.analyze(
                state.standalone_question,
                sources=self._known_sources(),
            )
            for tr in analysis.llm_traces:
                recorder.record_llm(
                    "brain",
                    tr.get("messages", []),
                    tr.get("completion", ""),
                    int(tr.get("latency_ms", 0)),
                )
            state.intent = analysis.intent
            state.plan = analysis.plan
            state.needs_table = analysis.needs_table
            state.needs_calculation = analysis.needs_calculation
            self._apply_focus(state)
            if analysis.intent == Intent.OUT_OF_SCOPE:
                state.stop_reason = "out_of_scope"
            recorder.emit(
                "analyze",
                payload={
                    **analysis.to_dict(),
                    "brain_llm_calls": len(analysis.llm_traces),
                    "focus_source": state.focus_source,
                    "focus_page": state.focus_page,
                },
            )
            return state

        def retrieve(state: AgentState) -> AgentState:
            if not state.plan:
                return state
            state.round += 1
            per_query_k = int(state.top_k or self.acfg.get("top_k_per_subquery", 5))
            max_queries = int(self.acfg.get("max_subqueries_per_round", 3))
            max_final = int(self.acfg.get("max_final_evidence", 8))
            if state.intent == Intent.SUMMARY:
                max_final += self.summary_bonus

            pool: dict[int, dict] = {}
            hyde_text = ""
            for sq in state.plan[:max_queries]:
                decision = route_subquery(
                    sq,
                    intent=state.intent,
                    round_no=state.round,
                    hyde_cfg=self.hyde_cfg,
                    needs_table=state.needs_table,
                    needs_calculation=state.needs_calculation,
                    summary_evidence_bonus=self.summary_bonus,
                    keyword_bm25_first=self.keyword_bm25_first,
                )
                state.needs_table = state.needs_table or decision.needs_table
                state.needs_calculation = (
                    state.needs_calculation or decision.needs_calculation
                )
                if not decision.channels:
                    continue

                search_sq = replace(sq, channels=decision.channels)
                if decision.use_hyde and self.hyde is not None:
                    if not hyde_text:
                        hyde_text = self._generate_hyde(state, recorder)
                    search_sq.hyde_text = hyde_text

                hits, timings = self.adapter.search(search_sq, top_k=per_query_k)
                fallback_used = False
                if not hits and decision.fallback_channels:
                    fallback_used = True
                    fallback_sq = replace(
                        search_sq, channels=decision.fallback_channels
                    )
                    hits, timings = self.adapter.search(
                        fallback_sq, top_k=per_query_k
                    )

                self.chunk_types.annotate(hits)
                recorder.emit(
                    "route",
                    round=state.round,
                    payload={
                        "query": sq.text,
                        "filter": sq.filter_expr,
                        "decision": decision.to_dict(),
                        "fallback_used": fallback_used,
                        "hyde_used": bool(search_sq.hyde_text),
                    },
                )
                recorder.emit(
                    "retrieve",
                    round=state.round,
                    latency_ms=timings.get("dense_ms", 0)
                    + timings.get("bm25_ms", 0)
                    + timings.get("rerank_ms", 0),
                    payload={
                        "query": sq.text,
                        "filter": sq.filter_expr,
                        "channels": list(search_sq.channels),
                        "counts": {
                            "dense": timings.get("dense_hits", 0),
                            "bm25": timings.get("bm25_hits", 0),
                            "fused": timings.get("fused_hits", 0),
                        },
                        "timings": timings,
                        "top_hits": [
                            {
                                "source": h.get("source"),
                                "page": h.get("page"),
                                "chunk_type": h.get("chunk_type"),
                                "rrf_score": round(float(h.get("rrf_score", 0.0)), 6),
                                "rerank_score": h.get("rerank_score"),
                                "channels": h.get("channels", []),
                            }
                            for h in hits[:5]
                        ],
                    },
                )
                self._merge_hits(pool, hits, sub_query=sq.text)

            round_ids = set(pool.keys())
            state.round_new_ids = round_ids - set(state.evidence.keys())
            state.round_hit_count = len(round_ids)

            ordered = sorted(
                pool.values(),
                key=lambda h: float(h.get("rerank_score", h.get("rrf_score", 0.0))),
                reverse=True,
            )[:max_final]
            self._register_evidence(state, ordered, max_final=max_final)
            if not ordered:
                state.stop_reason = "no_evidence"
            recorder.emit(
                "merge",
                round=state.round,
                payload={
                    "kept": len(ordered),
                    "total_evidence": len(state.evidence),
                    "new_chunks": len(state.round_new_ids),
                    "round_hits": state.round_hit_count,
                    "chunk_types": count_types(ordered),
                    "chunks": [ev.to_dict() for ev in state.evidence.values()],
                },
            )
            return state

        def assess(state: AgentState) -> AgentState:
            novelty = len(state.round_new_ids) / max(1, state.round_hit_count)
            outcome = self.assessor.assess(
                question=state.standalone_question,
                sub_queries=state.plan,
                contexts=state.contexts,
                novelty=novelty,
            )
            if outcome.llm_trace:
                tr = outcome.llm_trace
                recorder.record_llm(
                    "brain",
                    tr.get("messages", []),
                    tr.get("completion", ""),
                    int(tr.get("latency_ms", 0)),
                )
            suff = outcome.sufficiency
            state.sufficiency = suff
            state.conflicts = list(suff.conflicts)

            recorder.emit(
                "assess",
                round=state.round,
                payload={
                    **suff.to_dict(),
                    "degraded": outcome.degraded,
                    "error": outcome.error,
                    "new_chunks": len(state.round_new_ids),
                    "round_hits": state.round_hit_count,
                },
            )

            if suff.decision == Decision.SUFFICIENT:
                state.stop_reason = state.stop_reason or "sufficient"
                return state

            if suff.decision == Decision.CONFLICT:
                state.stop_reason = state.stop_reason or "conflict"
                recorder.emit(
                    "stop",
                    round=state.round,
                    payload={"reason": state.stop_reason, "decision": "conflict"},
                )
                return state

            can_continue = (
                state.round < state.budget.max_rounds
                and state.budget.can_continue(recorder.usage, state.started_at)
            )
            if can_continue and novelty > state.budget.novelty_epsilon:
                return state  # 进入 rewrite 节点

            if not can_continue:
                state.stop_reason = (
                    "max_rounds"
                    if state.round >= state.budget.max_rounds
                    else "budget"
                )
            else:
                state.stop_reason = "no_new_evidence"
            if state.contexts and self.on_budget_exhausted == "partial":
                state.status = Status.PARTIAL
            recorder.emit(
                "stop",
                round=state.round,
                payload={
                    "reason": state.stop_reason,
                    "decision": suff.decision.value,
                    "can_continue": can_continue,
                },
            )
            return state

        def rewrite(state: AgentState) -> AgentState:
            outcome = self.rewriter.rewrite(
                question=state.standalone_question,
                plan=state.plan,
                sufficiency=state.sufficiency,
                contexts=state.contexts,
                sources=self._known_sources(),
            )
            if outcome.llm_trace:
                tr = outcome.llm_trace
                recorder.record_llm(
                    "brain",
                    tr.get("messages", []),
                    tr.get("completion", ""),
                    int(tr.get("latency_ms", 0)),
                )
            recorder.emit(
                "rewrite",
                round=state.round,
                payload={
                    "source": outcome.source,
                    "reason": outcome.reason,
                    "error": outcome.error,
                    "sub_queries": [sq.to_dict() for sq in outcome.plan],
                },
            )
            if outcome.plan:
                state.plan = outcome.plan
                state.rewrite_count += 1
                self._apply_focus(state)
            else:
                state.plan = []
                state.stop_reason = state.stop_reason or "no_new_subqueries"
                if state.contexts and self.on_budget_exhausted == "partial":
                    state.status = Status.PARTIAL
            return state

        def tools(state: AgentState) -> AgentState:
            if not self.tools_enabled:
                return state
            remaining = state.budget.max_tool_calls - recorder.usage.tool_calls
            if remaining <= 0:
                return state

            self._ensure_tables(recorder)
            calls, llm_trace = self.tool_planner.plan(
                state.standalone_question,
                needs_table=state.needs_table,
                needs_calculation=state.needs_calculation,
            )
            if llm_trace and llm_trace.get("messages"):
                recorder.record_llm(
                    "brain",
                    llm_trace.get("messages", []),
                    llm_trace.get("completion", ""),
                    int(llm_trace.get("latency_ms", 0)),
                )
                recorder.emit(
                    "tool_select",
                    round=state.round,
                    latency_ms=int(llm_trace.get("latency_ms", 0)),
                    status="ok" if not llm_trace.get("error") else "error",
                    payload={
                        "calls": [c.to_dict() for c in calls],
                        "error": llm_trace.get("error", ""),
                    },
                )

            for call in calls[:remaining]:
                result = self.tool_registry.execute(call.name, call.arguments)
                entry = {**result.to_dict(), "reason": call.reason}
                state.tool_results.append(entry)
                recorder.record_tool(
                    result.name,
                    result.latency_ms,
                    result.ok,
                    {
                        "reason": call.reason,
                        "arguments": call.arguments,
                        "error": result.error,
                        "output": result.output[:500],
                    },
                )

            if (
                state.sufficiency is not None
                and state.sufficiency.decision != Decision.SUFFICIENT
                and state.tool_results
            ):
                state.status = Status.PARTIAL
            recorder.emit(
                "tools",
                round=state.round,
                payload={
                    "requested": len(calls),
                    "executed": len(state.tool_results),
                    "ok": sum(1 for r in state.tool_results if r.get("ok")),
                    "needs_table": state.needs_table,
                    "needs_calculation": state.needs_calculation,
                },
            )
            return state

        def compose(state: AgentState) -> AgentState:
            has_tool = any(r.get("ok") for r in state.tool_results)
            if state.intent == Intent.CHITCHAT:
                mode = "direct"
                messages = direct_messages(state.standalone_question)
            else:
                if not state.contexts and not has_tool:
                    state.status = Status.REFUSED
                    state.stop_reason = state.stop_reason or "no_evidence"
                    state.answer = refusal_text("未检索到可用证据")
                    recorder.emit(
                        "compose",
                        status="refused",
                        round=state.round,
                        payload={"mode": "refused", "reason": state.stop_reason},
                    )
                    return state
                mode = "grounded"
                messages = grounded_messages(
                    state.standalone_question,
                    state.contexts,
                    tool_results=state.tool_results,
                    cite_evidence=self.citations_enabled,
                    conflicts=state.conflicts,
                )

            t0 = time.perf_counter()
            answer, llm_usage = generate(messages)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            recorder.record_llm(
                "answer", messages, answer or "", latency_ms, usage=llm_usage or None
            )

            if answer and answer.strip():
                state.answer = answer.strip()
                if state.status == Status.RUNNING:
                    state.status = Status.ANSWERED
                state.confidence = 0.5
                state.stop_reason = state.stop_reason or (
                    "chitchat" if mode == "direct" else "answered"
                )
            else:
                state.status = Status.REFUSED
                state.stop_reason = "empty_answer"
                state.answer = refusal_text("模型未返回有效答案")

            recorder.emit(
                "compose",
                round=state.round,
                latency_ms=latency_ms,
                payload={
                    "mode": mode,
                    "answer_chars": len(state.answer),
                    "confidence": state.confidence,
                },
            )
            return state

        def verify(state: AgentState) -> AgentState:
            if not self.citations_enabled:
                return state
            if state.status not in (Status.ANSWERED, Status.PARTIAL):
                # 拒答/错误状态没有可校验答案，避免给拒答文案生成引用
                return state
            outcome = self.verifier.verify(
                answer=state.answer,
                contexts=state.contexts,
                tool_results=state.tool_results,
                sufficiency=state.sufficiency,
                conflicts=state.conflicts,
                direct=state.intent == Intent.CHITCHAT,
                allow_llm=recorder.usage.brain.calls < state.budget.max_brain_calls,
            )
            if outcome.llm_trace and outcome.llm_trace.get("messages"):
                tr = outcome.llm_trace
                recorder.record_llm(
                    "brain",
                    tr.get("messages", []),
                    tr.get("completion", ""),
                    int(tr.get("latency_ms", 0)),
                )
            state.citations = outcome.citations
            state.citation_coverage = outcome.citation_coverage
            state.invalid_citations = outcome.invalid_citations
            state.confidence = outcome.confidence
            if (
                outcome.confidence < self.verifier.tau
                and state.status in (Status.ANSWERED, Status.PARTIAL)
            ):
                state.stop_reason = "low_confidence"
            recorder.emit(
                "verify",
                round=state.round,
                payload=outcome.to_dict(),
            )
            return state

        def refuse(state: AgentState) -> AgentState:
            reason = state.stop_reason or "insufficient_evidence"
            state.answer = refusal_text(reason)
            state.status = Status.REFUSED
            state.confidence = 0.0
            state.citations = []
            state.stop_reason = reason
            recorder.emit(
                "refuse",
                round=state.round,
                payload={"reason": reason},
            )
            return state

        def finalize(state: AgentState) -> AgentState:
            if state.status == Status.RUNNING:
                state.status = Status.ANSWERED
            state.stop_reason = state.stop_reason or "answered"
            recorder.emit(
                "finalize_node",
                payload={"status": state.status.value, "stop_reason": state.stop_reason},
            )
            return state

        return {
            "normalize": normalize,
            "analyze": analyze,
            "retrieve": retrieve,
            "assess": assess,
            "rewrite": rewrite,
            "tools": tools,
            "compose": compose,
            "verify": verify,
            "refuse": refuse,
            "finalize": finalize,
        }

    def _after_assess(self, state: AgentState) -> str:
        suff = state.sufficiency
        wants_tool = (
            self.tools_enabled
            and (state.needs_table or state.needs_calculation)
            and not state.tool_results
            and state.usage.tool_calls < state.budget.max_tool_calls
            and state.budget.can_continue(state.usage, state.started_at)
        )
        if suff is not None and suff.decision == Decision.CONFLICT:
            return "compose" if state.contexts else "refuse"
        if suff is not None and suff.decision != Decision.SUFFICIENT:
            if state.stop_reason in {"max_rounds", "budget", "no_new_evidence"}:
                return "compose" if state.status == Status.PARTIAL else "refuse"
            if wants_tool:
                return "tools"
            return "rewrite"
        if wants_tool:
            return "tools"
        return "compose" if state.contexts else "refuse"

    def _build_edges(self) -> dict:
        return {
            "normalize": lambda s: "analyze",
            "analyze": lambda s: (
                "compose"
                if s.intent == Intent.CHITCHAT
                else "refuse"
                if s.intent == Intent.OUT_OF_SCOPE or not s.plan
                else "retrieve"
            ),
            "retrieve": lambda s: "assess" if s.contexts else "refuse",
            "assess": self._after_assess,
            "rewrite": lambda s: (
                "retrieve" if s.plan else ("compose" if s.contexts else "refuse")
            ),
            "tools": lambda s: (
                "rewrite"
                if (
                    s.sufficiency is not None
                    and s.sufficiency.decision != Decision.SUFFICIENT
                    and not s.tool_results
                    and s.plan
                    and s.budget.can_continue(s.usage, s.started_at)
                )
                else "compose"
                if (s.contexts or any(r.get("ok") for r in s.tool_results))
                else "refuse"
            ),
            "compose": lambda s: "verify" if self.citations_enabled else "finalize",
            "verify": lambda s: (
                "refuse" if s.stop_reason == "low_confidence" else "finalize"
            ),
            "refuse": lambda s: "finalize",
            "finalize": lambda s: END,
        }

    @staticmethod
    def _register_evidence(
        state: AgentState, hits: list[dict], *, max_final: int = 8
    ) -> None:
        """跨轮累加证据：chunk_id 去重，E1..En 编号在首轮确定后保持不变。"""
        for h in hits:
            cid = int(h["id"])
            rerank_score = h.get("rerank_score")
            existing = state.evidence.get(cid)
            if existing is None:
                evidence = Evidence(
                    evidence_id=f"E{len(state.evidence) + 1}",
                    chunk_id=cid,
                    text=h.get("text", ""),
                    source=h.get("source", ""),
                    page=int(h.get("page", 0)),
                    chunk_idx=int(h.get("chunk_idx", -1)),
                    chunk_type=str(h.get("chunk_type") or "prose"),
                    dense_score=float(h.get("dense_score", 0.0)),
                    bm25_score=float(h.get("bm25_score", 0.0)),
                    rrf_score=float(h.get("rrf_score", 0.0)),
                    rerank_score=None if rerank_score is None else float(rerank_score),
                    channels=list(h.get("channels", [])),
                    round_found=state.round,
                    sub_query=h.get("sub_query", ""),
                )
                state.evidence[cid] = evidence
                continue
            existing.channels = sorted(
                set(existing.channels) | set(h.get("channels", []))
            )
            existing.dense_score = max(
                existing.dense_score, float(h.get("dense_score", 0.0))
            )
            existing.bm25_score = max(
                existing.bm25_score, float(h.get("bm25_score", 0.0))
            )
            existing.rrf_score = max(
                existing.rrf_score, float(h.get("rrf_score", 0.0))
            )
            if rerank_score is not None:
                old = existing.rerank_score
                existing.rerank_score = max(
                    float(old) if old is not None else float("-inf"),
                    float(rerank_score),
                )

        def score(ev: Evidence) -> float:
            return ev.rerank_score if ev.rerank_score is not None else ev.rrf_score

        ordered = sorted(state.evidence.values(), key=score, reverse=True)[:max_final]
        state.contexts = [
            {
                "evidence_id": ev.evidence_id,
                "id": ev.chunk_id,
                "text": ev.text,
                "source": ev.source,
                "page": ev.page,
                "chunk_idx": ev.chunk_idx,
                "chunk_type": ev.chunk_type,
                "rrf_score": ev.rrf_score,
                "rerank_score": ev.rerank_score,
                "channels": list(ev.channels),
                "sub_query": ev.sub_query,
            }
            for ev in ordered
        ]
        state.references = [
            {
                "evidence_id": ev.evidence_id,
                "source": ev.source,
                "page": ev.page,
                "chunk_idx": ev.chunk_idx,
                "chunk_type": ev.chunk_type,
                "snippet": ev.text[:200],
                "score": round(score(ev), 6),
                "channels": list(ev.channels),
            }
            for ev in ordered
        ]

    def _merge_hits(
        self, pool: dict[int, dict], hits: list[dict], *, sub_query: str = ""
    ) -> None:
        """按 chunk_id 去重合并多子查询命中，同块保留各通道最高分。"""
        self.chunk_types.annotate(hits)
        for h in hits:
            h["sub_query"] = sub_query
            cid = int(h["id"])
            if cid not in pool:
                pool[cid] = dict(h)
                continue
            old = pool[cid]
            old["channels"] = sorted(
                set(old.get("channels", [])) | set(h.get("channels", []))
            )
            for key in ("dense_score", "bm25_score", "rrf_score"):
                old[key] = max(float(old.get(key, 0.0)), float(h.get(key, 0.0)))
            if "rerank_score" in h:
                old["rerank_score"] = max(
                    float(old.get("rerank_score", 0.0)),
                    float(h["rerank_score"]),
                )

    def _apply_focus(self, state: AgentState) -> None:
        """把会话焦点补到未显式指定过滤条件的子查询上。"""
        if not state.focus_source and state.focus_page is None:
            return
        updated = []
        for sq in state.plan:
            changes: dict[str, Any] = {}
            if sq.source is None and state.focus_source:
                changes["source"] = state.focus_source
            if sq.page is None and state.focus_page is not None:
                changes["page"] = state.focus_page
            updated.append(replace(sq, **changes) if changes else sq)
        state.plan = updated

    def _remember_session(self, state: AgentState, recorder: TraceRecorder) -> None:
        """回答后写回会话：焦点取排序第一的引用，供下一轮指代消解。"""
        top = state.references[0] if state.references else {}
        sources = [
            str(r["source"]) for r in state.references if r.get("source")
        ][:10]
        evidence_ids = [
            str(r["evidence_id"]) for r in state.references if r.get("evidence_id")
        ][:8]
        try:
            session = self.sessions.append(
                state.session_id,
                state.question,
                state.answer[:2000],
                sources=sources,
                evidence_ids=evidence_ids,
                focus_source=str(top.get("source") or ""),
                focus_page=(
                    int(top["page"]) if top.get("page") is not None else None
                ),
            )
        except Exception as exc:  # 会话写回失败不影响本次答案
            logger.warning("session 写回失败: %s", exc)
            recorder.emit("session", status="error", payload={"error": str(exc)})
            return
        recorder.emit("session", payload=session.summary())

    def _known_sources(self) -> list[str] | None:
        """已入库 source 清单（BM25 就绪时），用于约束 brain 的 source 过滤。"""
        try:
            if not getattr(self.pipeline, "_bm25_ready", False):
                return None
            rows = getattr(self.adapter.bm25, "_meta", []) or []
            sources = sorted({str(r["source"]) for r in rows if r.get("source")})
            return sources[:50] or None
        except Exception:
            return None

    def _ensure_tables(self, recorder: TraceRecorder) -> None:
        """懒加载兜底：注册表为空时从 BM25 全量 chunk 解析表格工件。"""
        if len(self.tables):
            return
        rows = getattr(self.adapter.bm25, "_meta", []) or []
        if rows:
            self.tables.ensure_built(rows, self.tables_dir)
        recorder.emit(
            "table_extract",
            status="ok" if len(self.tables) else "skipped",
            payload={"tables": len(self.tables), "chunks": len(rows)},
        )

    def _generate_hyde(self, state: AgentState, recorder: TraceRecorder) -> str:
        """每轮最多一次 HyDE brain 调用；失败不影响主检索。"""
        if self.hyde is None:
            return ""
        if recorder.usage.brain.calls >= state.budget.max_brain_calls:
            recorder.emit(
                "hyde",
                status="skipped",
                round=state.round,
                payload={"reason": "brain_budget"},
            )
            return ""
        try:
            text, tr = self.hyde.generate(state.standalone_question)
        except Exception as exc:
            recorder.emit(
                "hyde",
                status="error",
                round=state.round,
                payload={"error": str(exc)},
            )
            return ""
        recorder.record_llm(
            "brain",
            tr.get("messages", []),
            tr.get("completion", ""),
            int(tr.get("latency_ms", 0)),
        )
        recorder.emit(
            "hyde",
            round=state.round,
            latency_ms=int(tr.get("latency_ms", 0)),
            payload={
                "chars": len(text),
                "preview": text[:120],
                "question": state.standalone_question,
            },
        )
        return text


def build_agent_runtime(cfg: dict, pipeline, llms: dict[str, dict]) -> AgentRuntime | None:
    """按配置构建 Agent；默认关闭时返回 None，保证旧服务零开销。"""
    if not (cfg.get("agent", {}) or {}).get("enabled", False):
        return None
    return AgentRuntime(cfg, pipeline, llms)
