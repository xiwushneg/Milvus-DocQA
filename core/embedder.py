"""向量化模块：本地模型 / 远程 API / Mock 三种实现，统一接口。

- LocalEmbedder : sentence-transformers 加载本地模型（推荐 BAAI/bge-m3，1024 维）
- ApiEmbedder   : OpenAI 兼容 /embeddings 接口（如 SiliconFlow 托管 bge-m3，免 GPU）
- MockEmbedder  : 确定性哈希词袋向量，仅用于单元/冒烟测试，不依赖任何模型下载
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    name: str = "base"

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化（入库路径）。"""

    def embed_query(self, text: str) -> list[float]:
        """单条查询向量化（检索路径）。"""
        return self.embed_documents([text])[0]


class LocalEmbedder(BaseEmbedder):
    """基于 sentence-transformers 的本地向量化（延迟导入，按需加载权重）。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        from sentence_transformers import SentenceTransformer  # 延迟导入

        self._model = SentenceTransformer(model_name, device=device)
        self._batch = batch_size
        self.name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            batch_size=self._batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]


class ApiEmbedder(BaseEmbedder):
    """OpenAI 兼容 /embeddings 接口客户端（分批调用）。"""

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        batch_size: int = 32,
    ):
        import httpx

        # 连接级自动重试：DNS 抖动 / 瞬时断网时最多重试 3 次
        self._http = httpx.Client(
            transport=httpx.HTTPTransport(retries=3),
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        self._model_name = model
        self._base = api_base.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._batch = batch_size
        self.name = f"api:{model}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            batch = [t[:6000] for t in texts[i : i + self._batch]]
            resp = self._http.post(
                f"{self._base}/embeddings",
                headers=self._headers,
                json={"model": self._model_name, "input": batch},
                timeout=120,
            )
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
        return out


class MockEmbedder(BaseEmbedder):
    """确定性哈希字符二元组向量（L2 归一化），测试专用。

    语义相近的文本共享大量二元组，因此能在冒烟测试中产生合理检索结果。
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"mock-hash-bigram-{dim}"

    @staticmethod
    def _tokens(text: str) -> list[str]:
        t = re.sub(r"\s+", "", text.lower())
        grams = [t[i : i + 2] for i in range(max(len(t) - 1, 0))]
        return grams or ([t] if t else [])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in self._tokens(t):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                v[h % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


def build_embedder(cfg: dict) -> BaseEmbedder:
    ecfg = cfg.get("embedding", {})
    provider = ecfg.get("provider", "local")
    if provider == "local":
        return LocalEmbedder(
            model_name=ecfg.get("model", "BAAI/bge-m3"),
            device=ecfg.get("device", "cpu"),
            batch_size=ecfg.get("batch_size", 32),
        )
    if provider == "api":
        key = os.environ.get(ecfg.get("api_key_env", "EMBEDDING_API_KEY"), "")
        if not key:
            raise RuntimeError(
                f"provider=api 需要设置环境变量 {ecfg.get('api_key_env', 'EMBEDDING_API_KEY')}"
            )
        return ApiEmbedder(
            model=ecfg.get("model", "BAAI/bge-m3"),
            api_base=ecfg.get("api_base", ""),
            api_key=key,
            batch_size=ecfg.get("batch_size", 32),
        )
    if provider == "mock":
        return MockEmbedder(dim=int(cfg.get("milvus", {}).get("dense_dim", 256)))
    raise ValueError(f"未知 embedding provider: {provider}")
