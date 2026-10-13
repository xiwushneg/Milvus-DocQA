# 可选拓展：领域 LoRA 微调（需本机 GPU）

本系统的主路径是 **RAG（无需训练）**。若论文需要“模型优化”实验章节，可对开源
大模型做 LoRA 轻量微调，提升领域问答风格。以下流程在**你的本机**（NVIDIA GPU，
显存 ≥ 16GB）执行，推荐使用 LLaMA-Factory。

## 1. 准备环境（本机）

```bash
conda create -n docqa-ft python=3.10 -y && conda activate docqa-ft
git clone https://github.com/hiyouga/LLaMA-Factory && cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

## 2. 构造训练数据（data/docqa_lora.json）

```json
[
  {
    "instruction": "基于以下参考资料回答问题。\n### 参考资料：\n{检索到的上下文}\n### 问题：\n{问题}",
    "input": "",
    "output": "标准答案（人工撰写，或由更强模型生成后人工校对）"
  }
]
```

建议 300-1000 条；可借助本系统 evaluation 流程筛出回答质量差的样例，针对性补充。

## 3. LoRA 训练

新建 `qwen2.5_lora_docqa.yaml`：

```yaml
### model
model_name_or_path: Qwen/Qwen2.5-7B-Instruct

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 8
lora_alpha: 16

### dataset
dataset: docqa_lora
template: qwen
cutoff_len: 2048

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 5.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true

### output
output_dir: saves/qwen2.5-docqa-lora
logging_steps: 10
save_steps: 100
```

启动训练：

```bash
llamafactory-cli train qwen2.5_lora_docqa.yaml
```

## 4. 合并权重并部署

```bash
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_name_or_path saves/qwen2.5-docqa-lora \
  --template qwen \
  --export_dir models/qwen2.5-docqa-merged
```

用 vLLM 部署合并后的模型：

```bash
vllm serve models/qwen2.5-docqa-merged --port 8001
```

然后把本项目 `config/config.yaml` 的 `llm.base_url` 改为
`http://localhost:8001/v1` 即可切换到微调后的模型。

## 5. 评估建议

微调前后各跑一次 `evaluation/eval_generation.py`，对比忠实度/相关性均值，
作为论文实验数据。

> 注意：微调并非必需。RAG + 混合检索 + 重排通常是性价比最高的优化路径，
> 请在论文中说明技术选型依据。
