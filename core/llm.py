"""大模型客户端：OpenAI 兼容 /chat/completions 协议。

同一客户端适配三种后端：Ollama（本地）/ vLLM（本地生产）/ 云端托管 API。
"""
from __future__ import annotations

import os

SYSTEM_PROMPT = (
    "你是一个严谨的中文文档问答助手。请仅依据【参考资料】回答用户问题，"
    "关键结论后标注引用编号，如 [1]。若资料不足以回答，请直接说明"
    "“根据现有资料无法回答”，不要编造。"
)

CONDENSE_PROMPT = (
    "你是查询改写助手。请结合对话历史，把【新问题】改写成一个不依赖上下文、"
    "指代明确的独立问题：把“它/这个/那种”等指代词替换为其具体指代对象，保留原意。"
    "只输出改写后的问题本身，不要任何解释、前缀或引号。"
)


def build_rewrite_messages(history: list[dict], question: str) -> list[dict]:
    """多轮对话指代消解：把历史 + 新问题组装成改写 prompt。"""
    lines = [
        f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:500]}"
        for m in history[-6:]
    ]
    user = "\n".join(lines) + f"\n\n【新问题】\n{question}"
    return [
        {"role": "system", "content": CONDENSE_PROMPT},
        {"role": "user", "content": user},
    ]


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] （来源: {h['source']} 第{h['page']}页）\n{h['text']}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[dict]) -> list[dict]:
    context = build_context(hits) or "（无检索结果）"
    user = f"【参考资料】\n{context}\n\n【问题】\n{question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class LLMClient:
    def __init__(self, lcfg: dict):
        import httpx

        # 连接级自动重试：DNS 抖动 / 瞬时断网时最多重试 3 次
        self._http = httpx.Client(
            transport=httpx.HTTPTransport(retries=3),
            timeout=httpx.Timeout(300.0, connect=10.0),
        )
        self._base = (lcfg.get("base_url") or "").rstrip("/")
        self._model = lcfg.get("model", "qwen2.5:7b")
        key = os.environ.get(lcfg.get("api_key_env", "LLM_API_KEY"), "")
        self._headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._temperature = float(lcfg.get("temperature", 0.1))
        self._max_tokens = int(lcfg.get("max_tokens", 1024))

    def chat(self, messages: list[dict]) -> str:
        resp = self._http.post(
            f"{self._base}/chat/completions",
            headers=self._headers,
            json={
                "model": self._model,
                "messages": messages,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def chat_stream(self, messages: list[dict]):
        """流式生成：按 OpenAI 兼容 SSE 协议逐段 yield 文本增量。

        注意：此处刻意用 requests 而非 httpx——实测 httpx/httpcore 的
        流式消费在并发下会触发服务端发送反压，拖慢本地生成队列
        （十并发首 token 47~60s vs requests 2~14s），非流式仍走 httpx。
        """
        import json

        import requests as _requests

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": True,
        }
        with _requests.post(
            f"{self._base}/chat/completions",
            headers=self._headers,
            json=payload,
            stream=True,
            timeout=(10, 300),
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []  # 部分 chunk 只含 usage，choices 为空
                text = choices[0].get("delta", {}).get("content") if choices else None
                if text:
                    yield text

    async def astream_chat(self, messages: list[dict]):
        """异步流式生成（SSE 逐段 yield）。

        供 FastAPI 异步端点消费：客户端断连时 asyncio 取消可干净关闭
        底层连接，不会像同步流那样遗留阻塞线程（僵尸）拖垮服务。
        """
        import json

        import httpx as _httpx

        if not hasattr(self, "_ahttp"):
            # 读超时 120s 指两次数据块之间的最大间隔（含服务端排队时间）
            self._ahttp = _httpx.AsyncClient(
                timeout=_httpx.Timeout(120.0, connect=10.0)
            )
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": True,
        }
        async with self._ahttp.stream(
            "POST",
            f"{self._base}/chat/completions",
            headers=self._headers,
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                text = choices[0].get("delta", {}).get("content") if choices else None
                if text:
                    yield text
