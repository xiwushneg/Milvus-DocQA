#!/usr/bin/env bash
# 重新冻结检索基线：在 180/40 的 116 块语料上，用加难版评测集重跑三种检索策略。
# 前置：start.bat 已启动 Docker + Milvus（http://localhost:19530/health 可用）。
# 用法：在 Git Bash 中  bash scripts/refreeze_baseline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=config/config_baseline116.yaml
EVAL=evaluation/eval_set.json

echo "===== 0. 前置检查 ====="
.venv/Scripts/python.exe -c "from pymilvus import connections; connections.connect(alias='t', host='localhost', port='19530'); print('Milvus OK')" || { echo "Milvus 未启动，请先运行 start.bat"; exit 1; }

echo "===== 1. 确认基线语料 116 块（独立 collection，不动生产库） ====="
.venv/Scripts/python.exe scripts/ingest.py --path scripts/demo_data --config "$CFG"

echo "===== 2. 重跑检索消融（dense / hybrid / hybrid+rerank） ====="
.venv/Scripts/python.exe evaluation/eval_retrieval.py \
  --eval-set "$EVAL" --config "$CFG" --top-k 5 \
  --out evaluation/results_rerank_repro_hard.json

echo "===== 3. 重跑重排策略（显式 --rerank 覆盖开关） ====="
.venv/Scripts/python.exe evaluation/eval_retrieval.py \
  --eval-set "$EVAL" --config "$CFG" --top-k 5 --rerank \
  --out evaluation/results_rerank_repro_hard_rerank.json

echo "===== 4. 查看结果 ====="
cat evaluation/results_rerank_repro_hard_rerank.json
echo ""
echo "人工核对：dense Recall@5 应明显低于 1.000，hybrid 应有可观测增益。"
echo "确认后把 hybrid+rerank 那行数字回填 results_rerank_ablation.json 并同步文档。"
