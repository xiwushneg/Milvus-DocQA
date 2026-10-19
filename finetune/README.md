# 领域 LoRA 微调方案（Qwen3-1.7B · RTX 5060 8GB 实测适配）

本系统的主路径是 **RAG（无需训练）**。本目录提供可选的"模型优化"实验：
用 QLoRA 对 **Qwen3-1.7B** 做领域微调，通过基座/微调对比实验支撑论文的
消融分析章节。方案针对 **8GB 显存** 调过参数，无需租用云 GPU。

> 为什么选 Qwen3-1.7B：RAG 场景下知识由检索注入，模型只做"基于资料的
> 抽取式生成"，小模型即可胜任；1.7B 的 4-bit 训练显存约 4-5GB，在 8GB
> 卡上留有安全余量；Qwen3 同尺寸中文能力强，且 LLaMA-Factory 支持最好。

## 目录结构

```
finetune/
├── README.md                 # 本文件（完整操作手册）
├── gen_train_data.py         # 训练数据生成：文档 → 分块 → 教师模型生成问答对
├── qwen3_lora_docqa.yaml     # LLaMA-Factory QLoRA 训练配置
├── data/
│   ├── dataset_info.json     # LLaMA-Factory 数据集注册表（dataset_dir 用）
│   └── docqa_lora.json       # 生成的训练集（gen_train_data.py 产物）
├── compare_ft.py             # 微调前后对比：同一份资料、同一批问题
├── serve_local.py            # 微调模型的 OpenAI 兼容本地服务（接回 RAG 用）
├── saves/                    # 训练产物（LoRA 适配器，不入库）
└── models/                   # 合并导出的完整模型（不入库）
```

## 流程总览

```
① 生成训练数据        ② 搭训练环境           ③ QLoRA 训练
项目 venv + API       finetune/.venv-ft      ~20-40 分钟 / 300 条
gen_train_data.py  →  LLaMA-Factory      →   llamafactory-cli train
                                                      │
⑥ 接回 RAG / 论文评估  ⑤ 微调前后对比         ④ 合并导出
serve_local.py     ←  compare_ft.py      ←   llamafactory-cli export
config.yaml 切换
```

---

## ① 生成训练数据（项目 venv）

训练样本的 prompt 结构与运行时 RAG **完全一致**（复用 `core/llm.py` 的
SYSTEM_PROMPT 和【参考资料】/【问题】格式），并自动混入约 15% 的
"资料外问题 → 拒答"负样本，让模型学会"不知道就说不知道"。

```powershell
cd C:\Users\Administrator\Documents\trae_projects\tizhong\milvus-docqa
.venv\Scripts\activate
python finetune\gen_train_data.py --path scripts\demo_data --per-chunk 2
```

- 输出：`finetune/data/docqa_lora.json`（Alpaca 格式，含 system 字段）
- **样本量要求：≥300 条**。`scripts/demo_data` 只有 2 篇文档，生成的少量
  样本仅用于**跑通全流程**；正式训练请把更多 PDF/DOCX/MD 放进一个目录，
  然后 `--path` 指向它重新生成
- 生成后**人工抽查 20%**：答案是否忠于资料、引用 [1] 是否合理、有无幻觉

## ② 搭建训练环境（独立 venv，不污染项目依赖）

```powershell
cd C:\Users\Administrator\Documents\trae_projects\tizhong\milvus-docqa\finetune
python -m venv .venv-ft
.venv-ft\Scripts\activate
python -m pip install -U pip

# ⚠️ RTX 5060 是 Blackwell 架构（sm_120），必须用 PyTorch ≥2.7 + CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[metrics]"       # torch 已装，不再装 [torch] extra
pip install fastapi uvicorn       # 供 serve_local.py 使用
```

国内下载模型需设镜像（训练终端里执行）：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

## ③ QLoRA 训练（约 20-40 分钟 / 300 条样本）

```powershell
cd C:\Users\Administrator\Documents\trae_projects\tizhong\milvus-docqa\finetune\LLaMA-Factory
llamafactory-cli train ..\qwen3_lora_docqa.yaml
```

关键配置（已在 `qwen3_lora_docqa.yaml` 中调好）：

| 参数 | 值 | 说明 |
|---|---|---|
| quantization_bit | 4 | QLoRA 4-bit（NF4），训练显存约 4-5GB |
| lora_rank / alpha | 8 / 16 | 可训练参数约 0.5%，产物仅几十 MB |
| batch × grad_accum | 2 × 8 | 有效 batch=16 |
| learning_rate | 1e-4 | LoRA 标准学习率 |
| cutoff_len | 1536 | 覆盖"资料+问题+答案"长度 |
| val_size | 0.05 | 留 5% 验证集，`training_loss.png` 可进论文 |

**OOM 兜底**（万一显存不足，按序降级）：batch 2→1 且 grad_accum 8→16；
cutoff 1536→1024；lora_rank 8→4。

训练正常表现：loss 从 ~2.5 降到 1.0 附近并趋平；验证 loss 与训练 loss
同向下降（若验证 loss 回升说明过拟合，把 epoch 降到 2）。

## ④ 合并导出

```powershell
llamafactory-cli export ^
  --model_name_or_path Qwen/Qwen3-1.7B ^
  --adapter_name_or_path ..\saves\qwen3-docqa-lora ^
  --template qwen3 ^
  --export_dir ..\models\qwen3-docqa-merged
```

## ⑤ 微调前后对比（论文实验数据）

```powershell
cd C:\Users\Administrator\Documents\trae_projects\tizhong\milvus-docqa
finetune\.venv-ft\Scripts\activate
python finetune\compare_ft.py
```

脚本用同一份参考资料 + 同一批问题（含一个资料外拒答题），先测基座、再挂
LoRA 测微调版，输出并排对比。可编辑脚本内 `DEFAULT_QUESTIONS` 换成自己的
问题集，结果整理成表格进论文，例如：

| 问题 | 基座 Qwen3-1.7B | +LoRA 微调 | 观察 |
|---|---|---|---|
| 事实型问题 | 回答冗长、偶有编造 | 简洁、带 [1] 引用 | 风格对齐 |
| 资料外问题 | 试图回答 | 明确拒答 | 幻觉抑制 |

## ⑥ 接回 RAG 系统（可选演示）

```powershell
# 终端 A：启动本地模型服务（微调版）
python finetune\serve_local.py --model finetune\models\qwen3-docqa-merged --port 8009
# 对照实验时改成: python finetune\serve_local.py --model Qwen/Qwen3-1.7B --port 8009

# 终端 B：把 config/config.yaml 的 llm 段改为
#   base_url: "http://127.0.0.1:8009/v1"
#   model: "qwen3-docqa-ft"
# 然后正常启动 uvicorn / streamlit，问答即走本地微调模型
```

跑系统评估：分别对"基座服务"与"微调服务"执行 `evaluation/eval_generation.py`，
记录指标均值对比（切换服务即切换实验组），两张表合并进论文。

---

## 常见问题

- **显存不够 / CUDA out of memory**：按 ③ 的兜底参数降级；确认没有其他程序占用 GPU（`nvidia-smi` 查看）
- **下载模型慢/失败**：确认 `$env:HF_ENDPOINT="https://hf-mirror.com"` 已设置
- **`llamafactory-cli` 不是内部命令**：确认激活了 `finetune\.venv-ft` 且在 LLaMA-Factory 目录内安装过 `pip install -e`
- **回答带 `<think>` 标签**：serve/compare 已用 `enable_thinking=False` 关闭思考模式；如自己调用，务必同样设置
- **训练数据格式**：Alpaca + system 字段，注册表在 `data/dataset_info.json`，
  通过 yaml 里 `dataset_dir` 指向本目录生效，无需改 LLaMA-Factory 自带文件

## 论文写作建议

1. 实验章节写"基座 vs LoRA 微调"对比（同基座、同检索链路，控制变量）
2. 附训练配置表（本 README 的表格）与 loss 曲线（saves 下的 training_loss.png）
3. 结论建议表述为：微调显著改善**回答风格对齐**（引用格式、拒答行为），
   而知识正确性仍主要依赖 RAG 检索质量——两者互补而非替代
