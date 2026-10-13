"""Streamlit Web 界面：文档上传、智能问答、来源引用展示。

启动: streamlit run webui/streamlit_app.py
"""
import streamlit as st
import requests

st.set_page_config(page_title="Milvus-DocQA 文档问答", page_icon="📚", layout="wide")

API = st.sidebar.text_input("后端 API 地址", value="http://localhost:8000")

st.title("📚 基于 Milvus 与开源大模型的文档检索问答系统")
st.caption(
    "RAG 全链路：文档解析 → 递归分块 → BGE-M3 向量化 → Milvus → "
    "混合检索(稠密+BM25, RRF 融合) → 开源大模型生成引用答案"
)


def show_refs(refs: list) -> None:
    for i, r in enumerate(refs, 1):
        st.markdown(
            f"**[{i}]** `{r['source']}` 第{r['page']}页 · score={r['score']} · "
            f"通道: {'/'.join(r.get('channels', []))}\n\n> {r['snippet']}"
        )


tab_chat, tab_docs = st.tabs(["💬 智能问答", "📁 文档管理"])

with tab_chat:
    if "history" not in st.session_state:
        st.session_state.history = []
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("refs"):
                with st.expander("📎 参考来源"):
                    show_refs(turn["refs"])
    q = st.chat_input("请输入你的问题…")
    if q:
        st.session_state.history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("检索并生成中…"):
                try:
                    resp = requests.post(f"{API}/api/chat", json={"question": q}, timeout=600)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    st.error(f"请求失败：{e}（请确认后端已启动：uvicorn app.main:app --port 8000）")
                    st.stop()
            st.markdown(data["answer"])
            st.caption(f"⏱ 总耗时 {data['total_ms']}ms · 分阶段 {data['timings']}")
        st.session_state.history.append(
            {"role": "assistant", "content": data["answer"], "refs": data["references"]}
        )
        with st.chat_message("assistant"):
            pass  # 下轮渲染历史时展示引用

with tab_docs:
    ups = st.file_uploader(
        "上传文档（PDF / DOCX / MD / TXT，可多选）",
        accept_multiple_files=True,
        type=["pdf", "docx", "md", "txt", "markdown"],
    )
    if st.button("开始导入", disabled=not ups) and ups:
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
        st.metric("已入库文本块", stats.get("total_chunks", 0))
        st.write("按文档统计：", stats.get("sources", {}))
    except Exception:
        st.warning("无法连接后端，请先启动：uvicorn app.main:app --port 8000")
