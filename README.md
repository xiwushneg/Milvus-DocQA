# Milvus-DocQA：基于 Milvus 与开源大模型的文档检索问答系统

毕业设计题目《基于 Milvus 与开源大模型的文档检索问答系统设计与实现》的完整可运行工程。

系统采用 **RAG（Retrieval-Augmented Generation，检索增强生成）** 架构：

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
                                        最终 Top-N 上下文 + 提问 ──▶ 开源大模型
                                                        （Qwen2.5 等）   │ │
                                        带引用编号的答案 + 参考来源 ◀────┘ │
                         └──────────────────────────────────────────────┘
```

**核心设计点（论文可展开的创新/工程亮点）**

1. 混合检索：稠密向量（语义）+ BM25（关键词精确匹配）互补，RRF 融合无需分数归一化；
2. 全链路本地化可选：Milvus Lite/Standalone + 本地 BGE-M3 + Ollama Qwen2.5，数据不出本机；
3. 引用可溯源：答案中的 [1][2] 标注对应来源文件、页码与片段；
4. 完整评估：检索 Recall@K / MRR + 生成端 LLM-as-Judge 忠实度/相关性打分脚本。

## 三种部署模式

| 模式 | Milvus | Embedding | 大模型 | 硬件要求 | 适合场景 |
|------|--------|-----------|--------|----------|----------|
| A 零GPU快速体验 | Milvus Lite（pip 自带，免安装） | SiliconFlow API（bge-m3） | SiliconFlow API（Qwen2.5-7B） | 普通电脑 | 联调/评审 |
| B 半本地 | Milvus Lite | 本地 BGE-M3（CPU 可跑） | SiliconFlow API | CPU 8GB+ | 日常开发 |
| C 全本地（答辩推荐） | Docker Milvus Standalone | 本地 BGE-M3 | Ollama + qwen2.5:7b | GPU 6GB+ 或 32GB 内存 | 演示/隐私场景 |

## 快速开始

### 0. 安装依赖

```bash
pip install -r requirements.txt            # 基础依赖（含 Milvus Lite）
pip install -r requirements-local.txt      # 仅模式 B/C 需要（本地模型）
```

### 1. 配置

复制 `.env.example` 为 `.env` 并按需填写（API 模式需填 Key；Ollama 本地模式无需 Key）。
关键配置在 `config/config.yaml`，常用项也可用环境变量覆盖（见 `.env.example` 注释）。

### 2. 导入示例文档

```bash
python scripts/ingest.py --path scripts/demo_data
```

### 3. 启动 API 服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# 文档: http://localhost:8000/docs
```

### 4. 启动 Web UI（另开终端）

```bash
streamlit run webui/streamlit_app.py
```

或直接命令行问答：

```bash
python scripts/chat_cli.py "Milvus 是什么？"
```

## 测试与评估

```bash
pytest -v                                   # 单元测试 + Milvus Lite 端到端冒烟
python evaluation/eval_retrieval.py         # 检索评估：Recall@K / MRR（对比 dense / hybrid / hybrid+rerank）
python evaluation/eval_generation.py        # 生成评估：LLM-as-Judge 忠实度与相关性
```

## 项目结构

```
milvus-docqa/
├── app/                  # FastAPI 服务（路由、配置加载、数据模型）
├── core/                 # 核心流水线
│   ├── loader.py         #   文档解析（PDF/DOCX/MD/TXT → 段落）
│   ├── splitter.py       #   递归字符分块（中文标点感知 + 重叠）
│   ├── embedder.py       #   向量化（本地 sentence-transformers / API / Mock）
│   ├── vectorstore.py    #   Milvus 封装（MilvusClient：Lite/Standalone 通用）
│   ├── bm25.py           #   BM25 稀疏检索（jieba 分词）
│   ├── fusion.py         #   RRF 倒数排名融合
│   ├── reranker.py       #   可选交叉编码器重排（bge-reranker-v2-m3）
│   ├── llm.py            #   大模型客户端（OpenAI 兼容：Ollama/vLLM/API）
│   └── pipeline.py       #   RAG 编排（入库 / 问答）
├── scripts/              # 入库脚本、命令行问答、示例文档
├── evaluation/           # 评估脚本（检索指标 / 生成打分 / 评测集模板）
├── tests/                # pytest 单元与端到端冒烟测试
├── webui/                # Streamlit 前端
├── config/config.yaml    # 全局配置
├── docker-compose.yml    # Milvus Standalone（模式 C）
└── finetune/README.md    # 可选拓展：LoRA 领域微调指南（需本机 GPU）
```

## 常见问题

- **Windows 上 Milvus Lite 不可用？** Milvus Lite 仅支持 macOS/Linux，Windows 请用 `docker-compose up -d` 启动 Standalone，并把 `config.yaml` 的 `milvus.uri` 改为 `http://localhost:19530`。
- **首次本地向量化很慢？** BGE-M3 约需下载 2GB 权重；无 GPU 建议模式 A（API 托管同款模型）。
- **如何换大模型？** 改 `config.yaml` 的 `llm.model` 与 `base_url` 即可，协议统一为 OpenAI 兼容。
- **数据规模上限？** Milvus Lite 面向原型（万级块）；更大规模切换 Standalone，BM25 内存索引可升级为 Milvus 2.5+ 原生 BM25 稀疏检索。

## 学术诚信提示

本项目为教学与毕设参考实现，请在理解代码的基础上结合自己的工作完成论文撰写与答辩，并遵守所在学校的学术规范。
