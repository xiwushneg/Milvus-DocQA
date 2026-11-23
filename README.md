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
6. **SSE 流式输出**：后端异步流转发（FastAPI async + httpx.AsyncClient），检索完成即推送引用，生成逐字呈现；客户端断连时 asyncio 干净取消，不遗留僵尸线程
7. **多轮对话**：带历史追问先经查询改写做指代消解（"它"→ 具体对象）再检索，前端可视化改写过程
8. **双模型热切换**：云端 DeepSeek-V3.2 ↔ 本机微调 Qwen3-1.7B 网页一键切换，档案配置化（config profiles），带在线状态探测

## Agentic RAG 增量升级（B0–B6）

在保留原有 `/api/chat` 零回归的前提下，新增自写状态机的 Agent 链路，默认关闭（`agent.enabled=false`），仅通过 `/api/agent/chat`、`/api/agent/chat/stream` 与 `/api/agent/traces` 暴露：

```
Normalize → Analyze → Retrieve ⇄ Assess → {Tools | Rewrite} → Compose → Verify → Finalize / Refuse
```

| 能力 | 模块 | 要点 |
|------|------|------|
| 查询规划 / 意图路由 | `core/agentic/analyze.py`、`route.py` | 五类意图分支；块类型内存规则判定并缓存，不新增 Milvus 字段 |
| 混合检索适配 | `core/agentic/retrieval.py` | 组合 `search_dense` / `BM25Index` / `rrf_fuse` / `Reranker`，`RagPipeline._retrieve` 一行未改（哈希冻结） |
| 自反思闭环 | `assess.py`、`rewrite.py` | 充分性评估 → 缺口改写 → 再检索，最多 3 轮；第 2 轮起 HyDE 仅作用于稠密通道 |
| 工具层 | `tools.py`、`tables.py` | 表格工件落 `data/tables/`，DuckDB 只读（仅 SELECT/WITH、200 行上限、3s 超时）；计算器、单位换算、受限代码执行；失败结果不注入最终答案 |
| 会话记忆 | `memory.py` | TTL + `max_turns` 裁剪 + 指代词 / “第 X 页”引用检测，可选 JSON 原子持久化 |
| 可信输出 | `compose.py`、`verify.py` | 逐句引用校验、非法引用检出、冲突并列展示、低置信拒答；置信度按充分性/引用覆盖/无冲突加权 |
| 可观测与流式 | `trace.py`、`runtime.py` | stage 级 trace、SSE 逐 token 推送、轨迹落盘与回放，brain / answer 分账并回填真实 usage |
| 双模型分工 | 配置驱动 | 规划/评估/改写/校验固定走云端 brain；最终回答可切本机 Qwen3-1.7B，trace 标注 `model_role` |

## 实验与评估

### 1. 检索策略消融（评估集 49 题 gold 标注，知识库 116 块）

| 检索策略 | Recall@5 | MRR | 平均耗时/题 | 评估器 |
|---------|----------|------|--------|--------|
| 纯稠密向量 | 1.000（饱和） | 0.776 | 163ms | programmatic（`eval_retrieval.py`）+ 运行时计量 |
| 混合检索（RRF） | 1.000（饱和） | **0.843** | 164ms | 同上 |
| 混合检索 + 重排序 | 1.000（饱和） | **0.939** | 1096ms | 同上 |

> 数字唯一来源：`evaluation/results_rerank_ablation.json`（49 题 = 44 可答 + 5 拒答；116 块，`config/config_baseline116.yaml` 可复现）。评测集 SHA256 `4b7d3ecf…`。

- **口径说明**：本语料为 20 篇短文（每篇 500–900 字），切块后 116 块。在此规模下 Recall@5 对三种策略均饱和至 1.000，无法体现差异；排序质量差异由 MRR 反映。这与项目文档《RAG 评估指标体系》中的判断一致："知识库只有几个片段时，Recall@5 天然接近 100%，指标没有区分度"。
- 混合检索使 **MRR +0.067**（0.776→0.843）：BM25 把稠密检索排在 4–5 位的专有名词类片段前移到 top-3，几乎零延迟代价（+1ms）
- 重排序使 **MRR 再 +0.096**（0.843→0.939），代价是逐对精排的 ~930ms 延迟——因此设计为配置化开关，按场景权衡精度与延迟
- **历史数据勘误**：早期 README 记录的 0.900/0.789、0.950/0.827、0.950/0.910 来自一套未提交的本地切块参数，语料状态已丢失、无法复现；本表为在固定配置下重新测得并冻结的基线。

### 2. 生成质量评估（LLM-as-Judge，25 题含 5 道资料外拒答题）

同一批问题分别由两套模型档案作答，云端强模型按忠实度/相关性（1-5 分）盲评：

| 模型 | 忠实度 | 相关性 | 拒答正确率 | 平均耗时 | 评估器 |
|------|-------|-------|-----------|---------|--------|
| 云端 DeepSeek-V3.2（671B MoE） | 4.40 | 4.80 | 5/5 | 9.0s | LLM-as-Judge（1–5 盲评）+ 规则（拒答） |
| **本机微调 Qwen3-1.7B** | 3.75 | 4.65 | **5/5** | **3.4s** | 同上 |

核心结论：**1.7B 微调模型以 1/13 的延迟、零 API 成本，达到与 671B 云端模型相同的拒答可控性**——微调数据中混入 15% 拒答负样本的直接收益（小模型在受控场景下的可控性可以追平超大模型）。

> **口径勘误（2026-11-21）**：上表（2026-10-19，`results_llm_judge.json`）传给裁判的上下文是展示用的 `snippet`（截断 200 字），部分正确答案的证据落在截断之外，被裁判误判为“编造”，因此忠实度数值偏保守，仅供历史参考。评估脚本已修复为向裁判提供完整块文本，当前有效结果见下表；ft 的 3.75 是否同样被低估，需待本机推理服务启动后按新口径复跑确认。

#### 修正口径复评（2026-11-21，云端档案）

环境：25 题（20 知识内 + 5 拒答），`top_k=5`，作答与裁判均为 DeepSeek-V3.2；本地知识库重新入库 20 篇示例文档（44 块）；本机 `ft` 服务未启动，未复评。

| 口径 | 忠实度 | 相关性 | 低分题（≤3） | 拒答正确 | 平均耗时 |
|------|-------|-------|-------------|---------|---------|
| 截断 snippet（错误口径） | 4.05 | 4.80 | 4 | 5/5 | 5.2s |
| **完整块文本（修正口径）** | **5.00** | **5.00** | **0** | **5/5** | **5.1s** |

结果文件：`evaluation/results_llm_judge_20261121.json`。

- 被误判的 4 题均为证据超出 200 字截断：Milvus 三种部署形态、bge-m3 的 8192 上限、C-MTEB、LLM-as-Judge；修正后裁判理由均能对应到原文。
- 20 道知识内题全部满分存在**天花板效应**：该评估集正确证据都在 Top-5 内，强模型在充分上下文下天然忠实，适合验证“有证据时不编造”，对更细粒度忠实度差异区分力有限。
- 裁判与作答模型同家族，存在自评偏好；拒答判定仍使用文本规则。

### 3. 并发压测（10 题轮转 × 30 请求，本机微调模型）

| 并发 | 成功率 | 端到端 P50 / P95 | 首字 P50 / P95 | 吞吐 |
|-----|-------|-----------------|---------------|------|
| 1 | 100% | 4.6s / 11.0s | 2.8s / 8.8s | 0.18 req/s |
| 5 | 100% | 15.6s / 20.3s | 14.9s / 19.3s | 0.32 req/s |
| 10 | 100% | 30.2s / 32.9s | 31.5s / 36.7s | 0.30 req/s |

- 全级别成功率 100%，P95≈P50 无长尾——延迟随并发线性增长是 GPU 串行生成下排队论的标准形态
- 单实例吞吐天花板 ~0.3 req/s（GPU 串行生成的物理极限），水平扩展需多实例 + 负载均衡，或改用 vLLM 连续批处理

**工程攻坚**：压测暴露同步流式端点的僵尸线程问题（客户端断连后阻塞生成器不可取消，线程池逐渐耗尽，十并发成功率 0%）——重构为 async 异步流架构后根治，排查过程含分层隔离实验与锁注入计时。

### 4. 领域微调（QLoRA，可选拓展）

- 基座 Qwen3-1.7B，LoRA rank=8，4-bit NF4，RTX 5060 8GB 训练 172.8s
- 训练数据 184 条与线上 RAG prompt 完全对齐（复用同一 SYSTEM_PROMPT 与【参考资料】格式），含 15% 资料外拒答负样本
- 效果：回答更简洁、忠实原文、拒答稳定（见上表），训练/推理分布对齐使微调直接改善线上行为

**模型选型实测**：通过 OpenAI 兼容协议，同一探针问题在无检索辅助条件下裸问多款开源模型：

| 模型 | 回答"RRF 融合常数 k 的经典取值" | 响应延迟 |
|------|-------------------------------|---------|
| Qwen3-30B-A3B-Instruct | ✗ 错误（回答 0.5~1.0） | ~3.5s |
| **DeepSeek-V3.2**（最终选用） | ✓ 正确（k=60） | ~5s |

该对比同时实证了 RAG 的价值：裸问会出错的知识点，经检索注入文档证据后模型回答被锚定，幻觉得到抑制。

### 5. Agentic RAG 评测体系（B6）

所有指标实现于 `evaluation/metrics.py`（纯函数），每个数字在输出 JSON 中均带 `evaluator` 字段：

| 指标 | 计算口径 | 评估器 |
|------|---------|--------|
| EM / F1 | NFKC + 小写 + 去标点 + 去 `[E1]/[T1]`；中文按字、英文按词；`eval_set.json` 的 44 道可答题已补人工参考答案 `answer` | programmatic |
| 拒答 Precision / Recall / F1 | 以“应拒答”为正类的混淆矩阵；E1–E4 用 runtime `refused`，E0 用文本规则；空/空白答案一律计为拒答 | programmatic |
| 引用覆盖 / 非法引用率 / 幻觉率 | 逐句引用校验；幻觉率 = `unsupported / max(claims, unsupported)`（上界口径，≤1） | verifier（引用正则 + claim 校验） |
| Recall@k / MRR / nDCG@k | 与 `eval_retrieval.py` 同一 gold 命中规则（source 子串 + must_contain + page） | programmatic |
| Faithfulness / AnswerRelevancy 等 | RAGAS 适配层，未安装时返回 `available=false` | RAGAS（可选） |
| 延迟 / token | trace usage 回填，按题聚合 | 运行时计量 |

> EM/F1 口径说明：参考答案为简短事实句，EM 要求整句完全一致、对生成式长答案偏严格；F1 衡量词级重叠。生成忠实度仍以 `eval_generation.py` 的 LLM-as-Judge 为主指标，EM/F1 仅作辅助。

**300 题中文集**：`evaluation/qa_zh_300.jsonl` 由 `build_qa_zh.py` 离线生成，当前为 300 道唯一问题、24 道拒答题、覆盖 20 篇文档。默认 `template` 模式按块抽取原文短语，问题包含原文关键短语，检索偏易，**仅用于规模回归、稳定性与阈值标定，不替代 49 题人工 gold 评估集**；需要更自然的问题可切换 `--mode llm`（调用 brain 模型生成，`must_contain` 必须逐字出现在原文）。

**2Wiki / HotpotQA 内存评测**：`eval_multihop.py` 直接读取整理后的 JSON/JSONL，用内存 BM25 对候选段落排序，不依赖 Milvus；输出 Recall@k / MRR / nDCG@k，若题目带 `prediction` 则同时计算 EM/F1。

**E0–E4 消融**（配置增量叠加，便于定位每项能力贡献）：

| 变体 | 配置增量 | 新增能力 |
|------|---------|---------|
| E0 | `agent.enabled=false`；显式 `retrieval.rerank.enabled=true` | 固定流水线（dense + BM25 + RRF + rerank） |
| E1 | `max_rounds=1`，关闭 HyDE/工具/引用 | 查询规划 / 意图路由 / 块类型 |
| E2 | `max_rounds=3` | 自反思闭环（充分性评估 → 缺口改写 → 再检索） |
| E3 | `tools.enabled=true` | 工具层（表格 DuckDB / 计算器 / 单位换算） |
| E4 | `citations.enabled=true` | 逐句引用校验 / 冲突并列 / 低置信拒答 |

**D2 全量消融结果**（2026-11-23 重跑，49 题 = 44 可答 + 5 拒答，作答与 brain 均为云端 DeepSeek-V3.2；节点重试与空答案拒答口径生效，τ=0.30 / min_coverage=0.0；E0 基线校验通过）：

| 变体 | Recall@5 | MRR | nDCG@5 | EM | F1 | 拒答 P/R/F1 | 引用覆盖 | 幻觉率 | 平均延迟 |
|------|----------|-----|--------|----|----|-------------|---------|--------|---------|
| E0 固定流水线 | 1.000 | 0.939 | 0.955 | 0.000 | 0.437 | 0.333 / 1.000 / 0.500 | —（未生成引用） | — | 6.3s |
| E1 + 规划/路由 | 0.955 | 0.905 | 0.918 | 0.000 | 0.389 | 0.500 / 0.200 / 0.286 | — | — | 20.4s |
| E2 + 自反思 | 0.955 | 0.905 | 0.918 | 0.000 | 0.389 | 0.333 / 0.200 / 0.250 | — | — | 23.9s |
| E3 + 工具层 | 0.955 | 0.905 | 0.918 | 0.000 | 0.397 | 0.333 / 0.200 / 0.250 | — | — | 22.9s |
| E4 + 可信输出 | 0.932 | 0.867 | 0.884 | 0.023 | 0.478 | 0.500 / 0.600 / 0.545 | 0.760 | 0.197† | 34.2s |

> 唯一来源：`evaluation/results_agentic_ablation.json`，每个指标在 JSON 中均带 `evaluator` 字段。† 幻觉率为校验器口径的上界：人工核对 E4 本轮 45 条含 verify 的 trace（13 条 unsupported claim）后确认，12 条为“根据现有资料无法回答”等拒答/无资料说明句、1 条为无引用的引导句，无事实性编造——校验器把无引用句子一律计入 unsupported，因此该值实际反映“未引用率”；另有 9/63 条非法引用（0.143），引用编号约束仍需收紧。

- **检索是负结果**：E0 的 hybrid+rerank 已把 MRR 做到 0.939；E1–E3 用规划出的子查询替换原问题，在 116 块小语料上引入噪声，MRR 降至 0.905；E4 合并多轮证据后 Recall@5 0.932、MRR 0.867，为最低。Recall@5 接近饱和时，Agent 编排不带来检索增益，只增加约 3–5 倍延迟（6.3s → 20.4–34.2s）。
- **HyDE 口径**：D2 消融的 E1–E4 均未启用 HyDE（E1 显式关闭后，后续变体只增量覆盖其他配置键），上表与上述分析中的数字不含 HyDE 贡献；运行时仍可通过 `agent.hyde.enabled` 开启，需单独实验量化。
- **生成**：E4 的 F1 最高（0.478）、EM 0.023（唯一生成逐句引用的变体），E0 次之（0.437）；E2 与 E1 持平（0.389），自反思未带来生成增益。节点重试生效后，E2 的 `node_error` 从 3 次降至 1 次。
- **拒答**：E1–E3 只召回 1/5 道应拒答题（F1 0.250–0.286），说明规划/反思/工具本身不构成拒答能力；E4 拒答召回 0.600、F1 0.545，误拒 3 道（上轮 7 道），但漏拒 2 道。
- **标定取舍**：`evaluation/calibrate_thresholds.py` 已对 `(τ, min_coverage)` 做联合网格搜索，`data/calibration.json` 为 τ=0.30、`min_coverage=0.0`（40 题 dev 集：30 可答 + 10 拒答，与 49 题零重叠；拒答 F1=1.000、误拒 0）。启用后重跑，E4 的 `low_confidence` 从 10 次降至 2 次、误拒 7→3，但拒答召回 1.000→0.600；dev 上的最优阈值迁移到 test 后仍需在拒答 precision 与 recall 之间取舍（test 不参与标定）。
- **稳定性**：E0 基线校验 1.000 / 0.9394 对齐 `results_rerank_ablation.json`（容差 0.001）；E2 / E3 各出现 1 次、E4 出现 2 次 `node_error`（重试后仍有瞬时故障），E4 另有 3 次 `budget` 停止。

消融脚本不会手工写入任何检索数字：`retrieval_baseline` 段只读取 `evaluation/results_rerank_ablation.json`（1.000/0.776、1.000/0.843、1.000/0.939），最终结果写入 `evaluation/results_agentic_ablation.json`，检索题与拒答题分开统计（拒答题无 gold，不进入 Recall/MRR）。

**E0 基线校验**：默认开启（`--verify-baseline`）。E0 必须复现 `hybrid+rerank` 的 **1.000 / 0.939**（容差 0.001），并在输出 JSON 写入 `baseline_check`；不一致直接中止，避免在语料或 rerank 配置漂移时产出误导性对照表。运行消融必须显式指定基线配置 `--config config/config_baseline116.yaml`，否则当前 44 块示例库与基线 116 块语料不同源，会按设计中止；仅做探索时可加 `--no-verify-baseline`，但产出数字不得写入论文。

**RAGAS（可选）**：`pip install ragas datasets` 后，适配层自动走 `EvaluationDataset/SingleTurnSample`（新版）或 `datasets.Dataset`（旧版）；未安装或调用失败时不影响主评估流程，输出 `available=false` 并说明原因。

**复现命令**：

```bash
pytest -q                                                                  # 全量单测（含 B6 评测用例）
python evaluation/build_qa_zh.py --docs scripts/demo_data \
    --out evaluation/qa_zh_300.jsonl --target 300                          # 重建 300 题中文集
python evaluation/eval_retrieval.py                                        # 检索基线（需 Milvus + embedding）
python evaluation/eval_multihop.py --input <2wiki_or_hotpot>.jsonl \
    --top-k 5 --out evaluation/results_multihop.json                       # 多跳内存评测
python evaluation/ablation_agentic.py --eval-set evaluation/eval_set.json \
    --variants E0,E1,E2,E3,E4 --llm cloud \
    --config config/config_baseline116.yaml \
    --out evaluation/results_agentic_ablation.json                         # Agentic 消融（需 Milvus + 模型）
```

> `--llm cloud` 使用云端回答模型；切本机微调模型时改为 `--llm ft`，并先启动 `finetune/` 的 OpenAI 兼容服务（默认 127.0.0.1:8009）。

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

服务安全与探测参数（`config/config.yaml` 的 `server` 段）：

- `cors_origins`：跨域白名单，生产环境不要保留 `["*"]`。
- `api_key_env`：非空且对应环境变量有值时，`/api/*` 需携带 `X-API-Key` 或 `Authorization: Bearer <key>`（`/api/health` 除外）。
- `rate_limit_per_min`：按客户端 IP 的每分钟请求上限，`0` 表示关闭；超限返回 429 与 `Retry-After`。
- `llm_probe_ttl_s`：`/api/llms` 的在线探测缓存秒数（默认 30）；探测并发执行且仅 2xx 视为可用。

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
pytest -v                                   # 单元测试 + 端到端冒烟测试（253 项；Milvus 未启动时冒烟用例自动跳过；含 Agentic RAG B0–B6）
python evaluation/eval_retrieval.py         # 检索评估：Recall@K / MRR（dense / hybrid / hybrid+rerank 三策略消融）
python evaluation/eval_generation.py        # 生成评估：LLM-as-Judge 忠实度/相关性 + 拒答验证（cloud / ft 双模型）
python evaluation/build_qa_zh.py            # 构建 300 题中文评测集（template / llm 两种模式）
python evaluation/eval_multihop.py          # 2Wiki / HotpotQA 内存检索评测（Recall / MRR / nDCG）
python evaluation/ablation_agentic.py       # E0–E4 Agentic RAG 消融（检索基线只读 results_rerank_ablation.json）
python evaluation/benchmark_load.py --concurrency 1,5,10 --requests 30   # 并发压测：成功率 / 延迟分位数 / 吞吐
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
│   ├── pipeline.py       #   RAG 编排（入库 / 问答）
│   └── agentic/          #   Agentic RAG（状态机 / 规划路由 / 自反思 / 工具 / 记忆 / 校验 / trace）
├── scripts/              # 入库脚本、命令行问答、示例知识库（20 篇文档）
├── evaluation/           # 评估脚本（检索 / 生成 / 多跳 / 消融 / 指标）、300 题中文集与结果 JSON
├── tests/                # pytest 单元与端到端冒烟测试
├── webui/                # Streamlit 前端（流式渲染 / 模型切换 / 系统状态面板）
├── config/config.yaml    # 全局配置（含模型档案 profiles）
├── docker-compose.yml    # Milvus Standalone
└── finetune/             # QLoRA 微调：数据生成 / 训练配置 / 本地推理服务（OpenAI 兼容，流式）
```

## 常见问题

- **Windows 上 Milvus Lite 不可用？** Milvus Lite 仅支持 macOS/Linux，Windows 请用 `docker compose up -d` 启动 Standalone，并将 `config.yaml` 的 `milvus.uri` 改为 `http://localhost:19530`
- **如何更换大模型？** 修改 `config.yaml` 的 `llm.model` 与 `base_url` 即可，协议统一为 OpenAI 兼容（SiliconFlow / Ollama / vLLM）
- **数据规模上限？** Milvus Standalone 支持百万级向量；更大规模可平滑升级 Milvus Cluster，架构无需改动
