"""本地推理服务：把微调后的模型包装成 OpenAI 兼容 /v1/chat/completions 接口。

供本项目 config.yaml 直接切换（llm.base_url 指向本服务），无需安装 Ollama/vLLM。
Qwen3 默认关闭思考模式（enable_thinking=False），与 RAG 运行时行为一致。

用法（在 finetune/.venv-ft 虚拟环境中）:
    python finetune/serve_local.py --model finetune/models/qwen3-docqa-merged --port 8009
    python finetune/serve_local.py --model Qwen/Qwen3-1.7B --port 8009      # 基座对照
"""
import argparse
import time
import uuid

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    temperature: float = 0.1
    max_tokens: int = 512


app = FastAPI(title="DocQA Fine-tuned LLM Server")
STATE: dict = {}


@app.get("/v1/models")
def models():
    return {"data": [{"id": STATE["model_name"], "object": "model"}]}


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    tok, model = STATE["tok"], STATE["model"]
    text = tok.apply_chat_template(
        [m.model_dump() for m in req.messages],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tok(text, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=req.temperature > 0,
            temperature=max(req.temperature, 0.01),
            pad_token_id=tok.eos_token_id,
        )
    content = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    elapsed = time.time() - t0
    return {
        "id": f"chat-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": STATE["model_name"],
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(out.shape[1] - inputs["input_ids"].shape[1]),
        },
        "elapsed_ms": round(elapsed * 1000),
    }


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="微调模型本地 OpenAI 兼容服务")
    ap.add_argument("--model", required=True, help="合并后模型目录，或 HF 模型名（如 Qwen/Qwen3-1.7B）")
    ap.add_argument("--adapter", default=None, help="可选：不合并直接挂 LoRA 适配器目录")
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--cpu", action="store_true", help="强制 CPU 推理（默认自动用 GPU）")
    ap.add_argument(
        "--quant", action="store_true",
        help="4-bit 量化加载（省显存约 2GB 但生成慢 30%%；默认 bf16 更快更准）",
    )
    args = ap.parse_args()

    device = "cpu" if args.cpu else "auto"
    quant_kwargs = {}
    if args.quant:
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quant_kwargs["torch_dtype"] = torch.bfloat16  # 显式指定，避免默认加载成 fp32 占 7GB
    print(f"加载模型: {args.model}（{'4-bit' if args.quant else 'bf16'}, device={device}）…")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map=device, trust_remote_code=True, **quant_kwargs,
    )
    if args.adapter:
        print(f"挂载适配器: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    STATE.update(tok=tok, model=model, model_name=args.model.replace("\\", "/").rstrip("/"))
    print(f"就绪：http://127.0.0.1:{args.port}/v1")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
