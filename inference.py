from functools import lru_cache

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "alvaro-mazcu/Qwen3-4B-Instruct-FineTome"


@lru_cache(maxsize=2)
def load_model(model_id: str = DEFAULT_MODEL_ID, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def respond(message, history, temperature, max_tokens):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(DEFAULT_MODEL_ID, device)

    chat_history = []
    for turn in history:
        role = turn.get("role")
        content_blocks = turn.get("content") or []
        text_parts = [c.get("text", "") for c in content_blocks if isinstance(c, dict)]
        text = "\n".join([t for t in text_parts if t])
        if role and text:
            chat_history.append({"role": role, "content": text})
    chat_history.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return response


def build_app():
    with gr.Blocks(title="Qwen3-4B-Instruct-FineTome") as demo:
        gr.Markdown(
            "<h1 style='text-align:center;'>Qwen3-4B-Instruct-FineTome</h1>"
            "<p style='text-align:center;'>Chat with your fine-tuned model. Adjust sampling below.</p>"
        )

        chatbot = gr.Chatbot(height=520, label="Chat")
        temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
        max_tokens = gr.Slider(32, 512, value=256, step=16, label="Max New Tokens")

        gr.ChatInterface(
            fn=lambda message, history, temperature, max_tokens: respond(message, history, temperature, max_tokens),
            chatbot=chatbot,
            additional_inputs=[temperature, max_tokens],
            title=None,
            description=None,
            submit_btn="Send",
            stop_btn="Stop",
            fill_height=True,
        )
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
