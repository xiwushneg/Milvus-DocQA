# Milvus-DocQA

基于 **Milvus** 与**开源大模型**的文档检索问答系统 —— 一个完整可运行的 RAG（Retrieval-Augmented Generation，检索增强生成）工程实现。

> 导入 PDF / Word / Markdown 文档，用自然语言提问，获得**基于文档内容、带页码引用溯源**的答案。

个人独立完成，含自动化测试与量化评估体系：[GitHub 仓库](https://github.com/xiwushneg/Milvus-DocQA)

## 系统架构

```
                         ┌──────────────────────────────────────────────┐
                         │                 离线索引阶段                  │
 PDF/DOCX/MD/TXT ──解析──▶ 文本分块(递归字符+重叠) ──▶ BGE-M3 向量化 ──▶ │
                         │                                    Milvus    │
                         └──────────────────────────────────────────────┘
                         ┌──────────────────────────────────────────────┐
                         │                 在线问答阶段                  │
 用户提问 ──┬─ 稠密检索: Milvus 向量召回 Top-K ─────────────────────────┐ │
            └─ 稀疏检索: BM25 关键词召回 Top-K ──▶ RRF 倒数排名融合 ──┬─┘ │
                                                        （可选重排序）  │ │
                                        最终 Top-N 上下文 + 提问 ──▶ DeepSeek-V3.2
                                        带引用编号的答案 + 参考来源 ◀────┘ │
                         └──────────────────────────────────────────────┘
```

## 核心设计

1. **混合检索 + RRF 融合**：稠密向量（语义匹配）与 BM25（关键词精确命中）双通道召回，RRF 基于排名融合，规避两路分数量纲不一致的归一化问题，弥补单路检索的系统性盲区
2. **漏斗式检索架构**：双路各召回 Top20 → RRF 融合 → 可选交叉编码器精排（bge-reranker-v2-m3）→ Top5 上下文，逐层权衡召回率、排序精度与延迟
3. **引用可溯源**：分块保留页码元数据，答案中 [1][2] 标注对应来源文件、页码与片段，答案可验证
4. **生成层解耦**：嵌入与大模型均走 OpenAI 兼容协议，配置驱动，SiliconFlow 托管 / Ollama 本地 / vLLM 自建可一键切换
5. **评估驱动**：自建评估集，Recall@K / MRR 量化每一项检索优化决策，而非凭感觉调参

## 性能评测

评估集：20 题（gold 标注），知识库：20 篇技术文档 / 116 个分块，实测结果：

| 检索策略 | Recall@5 | MRR |
|---------|----------|------|
| 纯稠密向量检索 | 1.000 | 0.868 |
| 混合检索（RRF） | 1.000 | **0.902** |
| 混合检索 + 重排序 | 0.950 | **0.917** |

- 混合检索在召回不变的前提下将 MRR 提升 **3.4 个百分点**（排序质量显著改善）
- 重排序进一步提升头部精度，但 Recall@5 由 1.000 降至 0.950（精排模型将个别标准片段挤出前 5），因此将其设计为**配置化开关**，按场景权衡精度与完整性

**模型选型实测**：通过 OpenAI 兼容协议，同一探针问题在无检索辅助条件下裸问多款开源模型：

| 模型 | 回答"RRF 融合常数 k 的经典取值" | 响应延迟 |
|------|-------------------------------|---------|
| Qwen3-30B-A3B-Instruct | ✗ 错误（回答 0.5~1.0） | ~3.5s |
| **DeepSeek-V3.2**（最终选用） | ✓ 正确（k=60） | ~5s |

该对比同时实证了 RAG 的价值：裸问会出错的知识点，经检索注入文档证据后模型回答被锚定，幻觉得到抑制。

## 快速开始

### 0. 环境准备

- Python 3.12+
- Docker Desktop（运行 Milvus）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

复制 `.env.example` 为 `.env`，填入 SiliconFlow API Key（嵌入与生成模型均托管于 SiliconFlow，免费额度即可体验）。
检索与模型参数在 `config/config.yaml` 中调整。

### 3. 一键启动（Windows）

```bash
start.bat    # 启动 Docker Milvus → 后端 API(:8000) → WebUI(:8501) → 自动打开浏览器
stop.bat     # 一键停止
```

手动方式：

```bash
docker compose up -d                                  # Milvus Standalone
python scripts/ingest.py --path scripts/demo_data     # 导入示例文档
uvicorn app.main:app --port 8000                      # 后端 API，文档: http://localhost:8000/docs
streamlit run webui/streamlit_app.py                  # Web UI，http://localhost:8501
```

或直接命令行问答：

```bash
python scripts/chat_cli.py "什么是混合检索？"
```

## 测试与评估

```bash
pytest -v                                   # 单元测试 + 端到端冒烟测试（12 项）
python evaluation/eval_retrieval.py         # 检索评估：Recall@K / MRR（dense / hybrid / hybrid+rerank 三策略对比）
python evaluation/eval_generation.py        # 生成评估：LLM-as-Judge 忠实度与相关性
```

## 项目结构

```
milvus-docqa/
├── app/                  # FastAPI 服务（路由、配置加载、数据模型）
├── core/                 # 核心流水线
│   ├── loader.py         #   文档解析（PDF/DOCX/MD/TXT → 段落）
│   ├── splitter.py       #   递归字符分块（中文标点感知 + 重叠）
│   ├── embedder.py       #   向量化（API / 本地 sentence-transformers / Mock）
│   ├── vectorstore.py    #   Milvus 封装（Lite/Standalone 通用）
│   ├── bm25.py           #   BM25 稀疏检索（jieba 分词）
│   ├── fusion.py         #   RRF 倒数排名融合
│   ├── reranker.py       #   可选交叉编码器重排（bge-reranker-v2-m3）
│   ├── llm.py            #   大模型客户端（OpenAI 兼容协议）
│   └── pipeline.py       #   RAG 编排（入库 / 问答）
├── scripts/              # 入库脚本、命令行问答、示例知识库（20 篇文档）
├── evaluation/           # 评估脚本与评估集（20 题 gold 标注）
├── tests/                # pytest 单元与端到端冒烟测试
├── webui/                # Streamlit 前端
├── config/config.yaml    # 全局配置
├── docker-compose.yml    # Milvus Standalone
└── finetune/             # 可选拓展：LoRA 领域微调指南
```

## 常见问题

- **Windows 上 Milvus Lite 不可用？** Milvus Lite 仅支持 macOS/Linux，Windows 请用 `docker compose up -d` 启动 Standalone，并将 `config.yaml` 的 `milvus.uri` 改为 `http://localhost:19530`
- **如何更换大模型？** 修改 `config.yaml` 的 `llm.model` 与 `base_url` 即可，协议统一为 OpenAI 兼容（SiliconFlow / Ollama / vLLM）
- **数据规模上限？** Milvus Standalone 支持百万级向量；更大规模可平滑升级 Milvus Cluster，架构无需改动
