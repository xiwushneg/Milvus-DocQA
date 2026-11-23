# Milvus-DocQA D2 重跑与代码审查修复总结报告

- 日期：2026-11-23（重跑与提交）
- 提交：`22209a9`（master → origin/master）
- 结论：审查报告 A–F 批次修复完成；D2 全量消融已在 116 块基线上重跑，E0 基线校验通过；全量测试 253 项通过；工作区干净，已推送。

---

## 1. 摘要

1. **修复**：按审查报告完成 A–F 六批修复，覆盖 P0 真实 bug、接口/配置不一致、并发安全、规模化入库与工程收尾；`_retrieve` 源码哈希冻结与检索 parity 测试始终保持通过。
2. **重跑**：旧 D2 结果先备份，随后使用 `config/config_baseline116.yaml`（116 块独立集合）重跑 E0–E4，49 题（44 可答 + 5 拒答），作答与 brain 均为云端 DeepSeek-V3.2，启用节点重试、空答案拒答口径、联合标定 τ=0.30 / min_coverage=0.0。
3. **校验**：E0 复现 `hybrid+rerank` 基线 **Recall@5 1.000 / MRR 0.9394**（容差 0.001），`baseline_check.passed=true`。
4. **结果**：E4 生成 F1 0.478（各变体最高），拒答 P/R/F1 = 0.500/0.600/0.545，引用覆盖 0.760，幻觉率 0.197（校验器上界），平均延迟 34.2s。
5. **交付**：README 与设计文档数字已同步；`.gitignore` 与上传清单已整理；87 个文件提交并推送；测试 253 passed。

---

## 2. 代码审查修复回顾（批次 A–F）

| 批次 | 核心问题 | 主要改动 | 验证 |
|---|---|---|---|
| A（P0 低风险） | `state.usage` 恒为 0；工具 `[Tn]` 编号与 verify 错位；ingest 非法文件返回 500；流式 `node_error` 契约不一致；表注册表损坏导致启动失败；删除过滤转义缺失 | usage 回填同步；工具编号统一；`/api/ingest` 非法类型/损坏文件返回 4xx；流式改发 `error` 事件；注册表容错；过滤转义与空文件跳过原因 | 对应单测通过；接口回归通过 |
| B（并发） | BM25 重建与查询并发读写会导致新旧索引错配 | `BM25Index` 构建版本快照、锁内原子交换，读侧不中断；新增 1 写 N 读并发测试 | `test_bm25_concurrency` 通过；`test_pipeline_untouched`、`test_retrieval_parity` 通过 |
| C（口径/文档） | `--top-k` 未同时透传 E0 与 Agent；`--variants` 子集运行误报基线错误；HyDE 口径未标注；存在未读取配置 | top-k 透传；子集运行标记 `baseline_check.skipped`；文档标注 HyDE 未进 D2；清理死配置 | 消融单测通过；未覆盖历史 D2 结果 |
| D（性能/规模化） | 单文件入库且每文件全量重建 BM25；分页窗口受限；无内容去重；DuckDB 每次重建 | 批量入库单次 BM25 重建；`query_iterator` 优先、失败回退 offset（上限 16384）；sha256 清单去重（原子落盘）；DuckDB 内存连接池 | 入库批处理、分页、去重、回滚单测通过 |
| E（工程收尾） | 节点异常无重试；HTTP/LLM 资源未统一释放；入库失败无回滚；阈值仅 τ 标定；服务无安全配置 | `agent.node_retries=1` 并记录 `node_retry`；`close/aclose` 与 lifespan 清理；按已插入主键回滚；`(τ, min_coverage)` 联合网格搜索；API Key、限流、CORS 白名单配置 | 全量测试通过；联合标定产物 `data/calibration.json`（τ=0.30、min_coverage=0.0，dev 40 题 F1=1.000、误拒 0） |
| F（收尾补丁） | `/api/llms` 同步探测阻塞；DuckDB 只读校验误拒含单引号字面量的合法 SQL；空答案口径不一致 | 异步并发探测 + TTL 缓存（仅 2xx 可用）；先屏蔽单引号字面量再做危险语句校验；空/空白答案统一计为拒答 | 新增单测通过；全量 253 passed |

> 约束保持：核心模块修改未改变 `_retrieve` 行为，源码哈希冻结与检索 parity 测试持续通过；D2 重跑前旧结果已备份，未使用 `--no-verify-baseline` 规避校验。

---

## 3. D2 重跑方案

### 3.1 运行配置

| 项 | 取值 |
|---|---|
| 评测集 | `evaluation/eval_set.json`，49 题 = 44 可答 + 5 拒答 |
| 语料/集合 | `scripts/demo_data` 20 篇，`docqa_chunks_baseline116`，116 块 |
| 配置 | `config/config_baseline116.yaml`（chunk 180/40，rerank 强制启用） |
| 模型 | 作答 `cloud`，brain `cloud`（DeepSeek-V3.2） |
| 检索 | `top_k=5`，dense + BM25 + RRF + rerank |
| 可信输出 | τ_answer_confidence=0.30，min_coverage=0.0（联合标定产物） |
| 变体 | E0 / E1 / E2 / E3 / E4（配置增量叠加） |

### 3.2 执行命令

```bash
python evaluation/ablation_agentic.py \
  --eval-set evaluation/eval_set.json \
  --variants E0,E1,E2,E3,E4 \
  --llm cloud --top-k 5 \
  --config config/config_baseline116.yaml \
  --out evaluation/results_agentic_ablation.json
```

### 3.3 备份与基线校验

- 旧结果备份：`evaluation/results_agentic_ablation.before_d2rerun_20261123_2025.json`
- E0 基线校验：`expected = {recall_at_k: 1.000, mrr: 0.939}`，`actual = {1.000, 0.9394}`，容差 0.001，**通过**。
- 执行过程提示：首次启动误用了默认 44 块生产配置，启动后即停止（无结果产出），随后改用 `config_baseline116.yaml` 重新运行，最终结果与备份文件均未受影响。

---

## 4. 重跑结果

### 4.1 E0–E4 主表（2026-11-23 重跑）

| 变体 | Recall@5 | MRR | nDCG@5 | EM | F1 | 拒答 P/R/F1 | 引用覆盖 | 幻觉率 | 平均延迟 |
|---|---|---|---|---|---|---|---|---|---|
| E0 固定流水线 | 1.000 | 0.939 | 0.955 | 0.000 | 0.437 | 0.333 / 1.000 / 0.500 | —（未生成引用） | — | 6.3s |
| E1 + 规划/路由 | 0.955 | 0.905 | 0.918 | 0.000 | 0.389 | 0.500 / 0.200 / 0.286 | — | — | 20.4s |
| E2 + 自反思 | 0.955 | 0.905 | 0.918 | 0.000 | 0.389 | 0.333 / 0.200 / 0.250 | — | — | 23.9s |
| E3 + 工具层 | 0.955 | 0.905 | 0.918 | 0.000 | 0.397 | 0.333 / 0.200 / 0.250 | — | — | 22.9s |
| E4 + 可信输出 | 0.932 | 0.867 | 0.884 | 0.023 | 0.478 | 0.500 / 0.600 / 0.545 | 0.760 | 0.197 | 34.2s |

来源：`evaluation/results_agentic_ablation.json`，各指标带 `evaluator` 字段。

### 4.2 与修复前对比

| 指标 | 修复前 | 本次 | 变化说明 |
|---|---|---|---|
| E0 F1 | 0.415 | 0.437 | 空答案统一计拒答后略有变化 |
| E2 Recall@5 / MRR | 0.909 / 0.843 | 0.955 / 0.905 | 节点重试生效，失败轮次不再拖累检索 |
| E3 MRR | 0.894 | 0.905 | 同上，E1–E3 检索持平 |
| E4 MRR | 0.883 | 0.867 | 多轮证据合并后 Recall@5 0.955→0.932 |
| E4 F1 | 0.460 | 0.478 | 各变体最高 |
| E4 拒答 F1 | 0.588（P 0.417 / R 1.000） | 0.545（P 0.500 / R 0.600） | 误拒 7→3，但漏拒 0→2 |
| E4 `low_confidence` | 10 次 | 2 次 | 联合标定 min_coverage 0.60→0.0 生效 |
| E4 非法引用率 | 0.000 | 0.143（9/63） | 新增逐句校验暴露的短板 |
| E4 平均延迟 | 29.2s | 34.2s | 引用校验与节点重试增加开销 |
| `node_error` | E2/E3/E4 = 3/1/1 | 1/1/2 | 重试降低 E2 失败，E4 仍偶发 |

### 4.3 E4 trace 审计

- 本轮 E4 共有 45 条含 `verify` 事件的 trace（2 条 `out_of_scope`、2 条 `node_error` 在 verify 前结束）。
- 13 条 unsupported claim 全部人工复核：12 条为“根据现有资料无法回答”等拒答/无资料说明句，1 条为无引用的引导句，**未发现事实性编造**。
- 幻觉率 0.197 为校验器上界（无引用句一律计入 unsupported），实际反映“未引用率”。
- 63 条引用中 9 条非法（0.143），引用编号约束仍需收紧。

### 4.4 结论

- **检索仍是负结果**：E0 的 hybrid+rerank 已把 MRR 做到 0.939；E1–E3 用规划子查询替换原问题后降至 0.905，E4 合并多轮证据后降至 0.867。Recall@5 接近饱和时，Agent 编排不带来检索增益，延迟增加约 3–5 倍。
- **生成**：E4 F1 最高（0.478），E0 次之（0.437）；E1 与 E2 持平（0.389），自反思在小语料上没有生成增益。
- **拒答**：E1–E3 仅召回 1/5 道应拒答题；E4 召回 3/5、F1 0.545，误拒从 7 道降至 3 道，但出现 2 道漏拒。
- **标定取舍**：联合标定显著缓解了 `min_coverage=0.60` 造成的钳位误拒，但 dev 上的最优 τ 迁移到 test 后不能同时保证高 precision 与高 recall；test 全程未参与标定。
- **稳定性**：节点重试已生效（E2 `node_error` 3→1），但 E2/E3/E4 仍有 1/1/2 次瞬时失败；E4 另有 3 次 `budget` 停止。

---

## 5. 上传清单与工程状态

### 5.1 `.gitignore` 规则

入库：

- 评测集：`evaluation/eval_set.json`、`evaluation/dev_set.jsonl`、`evaluation/qa_zh_300.jsonl`
- 标定产物：`data/calibration.json`、`data/calibration_joint.json`、`data/dev_confidence*.jsonl`
- 正式结果：`evaluation/results_agentic_ablation.json`、`evaluation/results_rerank_ablation.json`、`evaluation/results_llm_judge_20261121.json`

排除：

- 中间/历史产物：`evaluation/results_rerank_repro*.json`、`evaluation/results_smoke.json`
- 备份：`evaluation/results_agentic_ablation.before_*.json`、`data/calibration.before_*.json`
- 内部资料：`.trae/`
- 运行时数据：`data/tables/`、`data/traces/`、`data/ingest_manifest.json`
- 已确认排除：`evaluation/eval_set_v1_20q.json`

### 5.2 提交与验证

- 提交：`22209a9 落地 Agentic RAG 全流程并重跑 D2 消融`
- 规模：87 files changed，+15251 / -172
- 测试：`pytest -q` → **253 passed**（Milvus 已启动，端到端冒烟用例实际执行）
- 文档：[README.md](../README.md) 与 [agentic-rag-design.md](agentic-rag-design.md) 的 D2 表、标定说明、测试计数均已同步
- 工作区：干净，`master` 与 `origin/master` 同步

---

## 6. 遗留问题与后续建议

1. **引用质量**：E4 非法引用 9/63（0.143）、覆盖率 0.760。建议收紧 `[En]` 编号校验，按 claim 重要性加权覆盖，或对拒答/无资料说明句豁免引用要求。
2. **拒答阈值迁移**：dev 上误拒为 0，但 test 上仍有 3 道误拒、2 道漏拒。若要偏保守可提高 τ；若要减少漏拒可降低 τ。建议扩大 dev 集或引入代价敏感阈值，并保持 test 不参与标定。
3. **节点稳定性**：重试已把 E2 的 `node_error` 从 3 次降至 1 次，但 E4 仍有 2 次。后续可加断路器/指数退避，或在预算耗尽前优先完成已获得证据的作答。
4. **HyDE 未进入 D2**：E1–E4 均未启用 HyDE，当前数字不含其贡献；若需量化收益，需单独重跑并备份现有结果。
5. **未产出正式结果的能力**：多跳评测（`eval_multihop.py` 已有脚本，缺少 2Wiki/HotpotQA 数据）、RAGAS（可选依赖未安装）、扫描 PDF OCR（缺少原生依赖）均未纳入本轮 D2。
6. **E0 拒答口径**：E0 依赖文本规则，本轮误拒 10 道；若论文对比需要，可对 E0 单独做规则阈值标定，但会改变基线口径，需谨慎。

---

## 附录：关键产物

| 产物 | 路径 |
|---|---|
| D2 结果 | `evaluation/results_agentic_ablation.json` |
| D2 旧结果备份 | `evaluation/results_agentic_ablation.before_d2rerun_20261123_2025.json` |
| 检索基线 | `evaluation/results_rerank_ablation.json` |
| 联合标定 | `data/calibration.json`、`data/calibration_joint.json` |
| dev 置信度 | `data/dev_confidence.jsonl`、`data/dev_confidence_joint.jsonl` |
| 基线配置 | `config/config_baseline116.yaml` |
| 修复前审查报告 | `.trae/documents/Milvus-DocQA-程序审查报告.md`（内部资料，未入库） |
