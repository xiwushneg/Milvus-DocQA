"""Streamlit Web 界面：文档上传、智能问答、来源引用展示。

启动: streamlit run webui/streamlit_app.py
"""
import html
import json

import streamlit as st
import requests

st.set_page_config(page_title="Milvus-DocQA 文档问答", page_icon="📚", layout="wide")

# ==================== 全局样式 ====================
st.markdown(
    """
<style>
  .hero {
    background: linear-gradient(135deg, #3b5bdb 0%, #7048e8 100%);
    border-radius: 16px; padding: 20px 28px; color: #fff; margin-bottom: 16px;
  }
  .hero-title { font-size: 1.4rem; font-weight: 700; letter-spacing: .3px; }
  .hero-sub { margin-top: 10px; }
  .pill {
    display: inline-block; background: rgba(255,255,255,.16);
    border: 1px solid rgba(255,255,255,.35); border-radius: 999px;
    padding: 2px 12px; margin: 4px 8px 0 0; font-size: .78rem;
  }
  .ref-card {
    background: #f8f9fc; border-left: 3px solid #4c6ef5; border-radius: 8px;
    padding: 10px 14px; margin: 8px 0; font-size: .88rem;
  }
  .ref-idx { color: #4c6ef5; font-weight: 700; margin-right: 4px; }
  .ref-meta { color: #868e96; font-size: .78rem; margin-left: 6px; }
  .ref-snip { color: #495057; margin-top: 6px; line-height: 1.5; }
  .doc-chip {
    display: inline-block; background: #edf2ff; border: 1px solid #dbe4ff;
    border-radius: 999px; padding: 3px 12px; margin: 3px 6px 3px 0;
    font-size: .8rem; color: #3b5bdb;
  }
  [data-testid="stSidebar"] { background: #f7f8fa; }
  .status-dot { margin-right: 6px; }
</style>
""",
    unsafe_allow_html=True,
)

# ==================== 侧边栏：连接 / 模型切换 / 系统状态 ====================
with st.sidebar:
    st.header("⚙️ 连接设置")
    API = st.text_input("后端 API 地址", value="http://localhost:8000")

    # ---- 生成模型切换器（档案列表来自后端 GET /api/llms） ----
    @st.cache_data(ttl=15)
    def _fetch_llms(api: str) -> list:
        return requests.get(f"{api}/api/llms", timeout=8).json()

    def _load_llms() -> list:
        try:
            return _fetch_llms(API)
        except Exception:
            return []

    LLMS = _load_llms()
    LLM_IDS = [m["id"] for m in LLMS] or ["cloud"]
    LLM_LABELS = {m["id"]: m["label"] for m in LLMS}
    LLM_AVAIL = {m["id"]: m.get("available", True) for m in LLMS}
    chosen_llm = st.radio(
        "生成模型",
        LLM_IDS,
        format_func=lambda x: ("🟢 " if LLM_AVAIL.get(x, True) else "🔴 ")
        + LLM_LABELS.get(x, x),
    )
    if LLMS and not LLM_AVAIL.get(chosen_llm, True):
        st.warning("所选模型服务未在线，提问前请先启动对应服务（如 finetune/serve_local.py）")

    st.divider()
    st.header("🖥 系统状态")
    backend_ok = milvus_ok = False
    try:
        requests.get(f"{API}/api/health", timeout=5).raise_for_status()
        backend_ok = True
        requests.get(f"{API}/api/documents", timeout=8).raise_for_status()
        milvus_ok = True
    except Exception:
        pass
    dot = lambda ok: "🟢" if ok else "🔴"
    st.markdown(
        f'{dot(backend_ok)}<span class="status-dot">后端 API（8000）</span><br>'
        f'{dot(milvus_ok)}<span class="status-dot">Milvus 知识库（19530）</span><br>'
        f'{dot(LLM_AVAIL.get("cloud", False))}<span class="status-dot">云端模型服务</span><br>'
        f'{dot(LLM_AVAIL.get("ft", False))}<span class="status-dot">本机微调服务（8009）</span>',
        unsafe_allow_html=True,
    )

# ==================== 主界面 ====================
st.markdown(
    """
<div class="hero">
  <div class="hero-title">📚 基于 Milvus 与开源大模型的文档检索问答系统</div>
  <div class="hero-sub">
    <span class="pill">RAG 全链路</span>
    <span class="pill">稠密 + BM25 混合检索</span>
    <span class="pill">RRF 融合</span>
    <span class="pill">双模型一键切换</span>
    <span class="pill">页码级引用溯源</span>
    <span class="pill">多轮对话</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


def show_refs(refs: list) -> None:
    """卡片式来源引用展示。"""
    for i, r in enumerate(refs, 1):
        channels = "/".join(r.get("channels", []))
        st.markdown(
            f'<div class="ref-card"><span class="ref-idx">[{i}]</span>'
            f"<b>{html.escape(r['source'])}</b> · 第{r['page']}页"
            f'<span class="ref-meta">score={r["score"]} · {html.escape(channels)}</span>'
            f'<div class="ref-snip">{html.escape(r["snippet"])}</div></div>',
            unsafe_allow_html=True,
        )


tab_chat, tab_docs = st.tabs(["💬 智能问答", "📁 文档管理"])

with tab_chat:
    if "history" not in st.session_state:
        st.session_state.history = []
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("llm_label"):
                st.caption(f"🤖 {turn['llm_label']}")
            if turn.get("refs"):
                with st.expander("📎 参考来源"):
                    show_refs(turn["refs"])
    q = st.chat_input("请输入你的问题…")
    if q:
        st.session_state.history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            status = st.empty()
            status.caption("🔍 检索资料中…")
            holder = st.empty()
            answer, refs, timings, total_ms, err = "", [], None, None, None
            recent = [
                {"role": t["role"], "content": t["content"]}
                for t in st.session_state.history[-7:-1]  # 排除刚追加的当前问题，带最近 3 轮
            ]
            try:
                resp = requests.post(
                    f"{API}/api/chat/stream",
                    json={"question": q, "llm": chosen_llm, "history": recent},
                    stream=True, timeout=600,
                )
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    ev = json.loads(line[5:].strip())
                    t = ev.get("type")
                    if t == "rewrite":
                        status.caption(f"🧠 指代消解 → {ev.get('query', '')}")
                    elif t == "refs":
                        refs = ev.get("references", [])
                        status.caption("✍️ 检索完成，生成中…")
                    elif t == "delta":
                        answer += ev.get("text", "")
                        holder.markdown(answer + "▌")
                    elif t == "done":
                        timings, total_ms = ev.get("timings"), ev.get("total_ms")
                    elif t == "error":
                        err = ev.get("detail", "未知错误")
            except Exception as e:
                st.error(f"请求失败：{e}（请确认后端已启动：uvicorn app.main:app --port 8000）")
                st.stop()
            if err:
                st.error(f"生成失败：{err}")
                st.stop()
            status.empty()
            holder.markdown(answer)
            model_label = LLM_LABELS.get(chosen_llm, chosen_llm)
            if refs:
                with st.expander("📎 参考来源"):
                    show_refs(refs)
            st.caption(f"🤖 {model_label} · ⏱ 总耗时 {total_ms}ms")
            if timings:
                with st.expander("⚡ 分阶段耗时"):
                    st.markdown(" · ".join(f"`{k}` {v}ms" for k, v in timings.items()))
        st.session_state.history.append(
            {"role": "assistant", "content": answer, "refs": refs, "llm_label": model_label}
        )
        with st.chat_message("assistant"):
            pass  # 下轮渲染历史时展示引用

with tab_docs:
    ups = st.file_uploader(
        "上传文档（PDF / DOCX / MD / TXT，可多选）",
        accept_multiple_files=True,
        type=["pdf", "docx", "md", "txt", "markdown"],
    )
    if st.button("开始导入", type="primary", disabled=not ups) and ups:
        files = [("files", (u.name, u.getvalue())) for u in ups]
        with st.spinner("解析 / 向量化 / 入库中…"):
            r = requests.post(f"{API}/api/ingest", files=files, timeout=1800)
        if r.ok:
            for d in r.json().get("documents", []):
                st.success(f"{d['source']}: {d['segments']} 段 → {d['chunks']} 块，{d['elapsed_ms']}ms")
        else:
            st.error(r.text)
    try:
        stats = requests.get(f"{API}/api/documents", timeout=10).json()
        col_metric, col_sources = st.columns([1, 3])
        with col_metric:
            st.metric("📚 已入库文本块", stats.get("total_chunks", 0))
        with col_sources:
            sources = stats.get("sources", {})
            if sources:
                chips = "".join(
                    f'<span class="doc-chip">{html.escape(k)} · <b>{v}</b></span>'
                    for k, v in sorted(sources.items())
                )
                st.markdown(f'<div style="margin-top:14px">{chips}</div>', unsafe_allow_html=True)
    except Exception:
        st.warning("无法连接后端，请先启动：uvicorn app.main:app --port 8000")
