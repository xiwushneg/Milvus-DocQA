"""微调前后对比脚本：同一份参考资料 + 相同问题，分别用"基座模型"和"基座+LoRA"生成回答。

直接加载 4-bit 基座，先测 before，再挂载 LoRA 适配器测 after，8GB 显存可跑。

用法（在 finetune/.venv-ft 虚拟环境中，项目根目录执行）:
    python finetune/compare_ft.py
    python finetune/compare_ft.py --adapter finetune/saves/qwen3-docqa-lora --q-count 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.llm import SYSTEM_PROMPT  # noqa: E402

DEFAULT_QUESTIONS = [
    "Milvus 为什么适合做向量检索？",
    "RAG 相比直接让大模型回答有什么优势？",
    "股票明天会涨吗？",  # 资料外问题，验证拒答能力
]

REF_TEXT = """Milvus 是一款开源向量数据库，专为海量向量数据的存储与检索设计。
它支持稠密向量、稀疏向量等多种索引类型，提供 HNSW 等高性能索引结构，
可部署为单机 Standalone 模式或分布式集群模式。在 RAG（检索增强生成）架构中，
Milvus 常被用作知识库底座：文档被切分为片段并向量化后存入 Milvus，
查询时将用户问题向量化并进行相似度检索，召回最相关的片段供大模型生成答案。
相比直接使用大模型闭卷回答，RAG 通过引用真实文档片段，可显著降低幻觉并支持答案溯源。"""


def load_model(base: str, adapter: str | None, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map=device, trust_remote_code=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def answer(tok, model, question: str, max_new_tokens: int = 384) -> str:
    import torch

    user = f"【参考资料】\n[1] （来源: demo.md 第1页）\n{REF_TEXT}\n\n【问题】\n{question}"
    text = tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Qwen3: 关闭思考模式，与 RAG 运行时一致
    )
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="微调前后回答质量对比")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default="finetune/saves/qwen3-docqa-lora")
    ap.add_argument("--q-count", type=int, default=3, help="取前几个默认问题")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch  # noqa: F401
    questions = DEFAULT_QUESTIONS[: args.q_count]

    print(f"加载基座模型: {args.base}（4-bit）…")
    tok, model = load_model(args.base, None, args.device)
    print("\n===== 微调前（基座） =====")
    before = [answer(tok, model, q) for q in questions]
    for q, a in zip(questions, before):
        print(f"\nQ: {q}\nA: {a or '（空输出）'}")

    adapter = Path(args.adapter)
    if not adapter.exists():
        print(f"\n[提示] 未找到适配器 {adapter}，跳过微调后对比。")
        return
    print(f"\n挂载 LoRA 适配器: {adapter} …")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter).eval()

    print("\n===== 微调后（基座 + LoRA） =====")
    for q, a_b in zip(questions, before):
        a_a = answer(tok, model, q)
        print(f"\nQ: {q}\nA(前): {a_b or '（空输出）'}\nA(后): {a_a or '（空输出）'}")

    print("\n提示：将以上对比结果截图/整理进论文实验章节；更多问题可自行追加到脚本 DEFAULT_QUESTIONS。")


if __name__ == "__main__":
    main()
