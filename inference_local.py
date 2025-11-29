import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(repo_id: str, device: str, token: str):
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, token=token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32,
        device_map=None,
        token=token,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def chat_loop(model, tokenizer, device: str):
    print("Chatting with the model. Type 'exit' to quit.")
    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        history.append({"role": "user", "content": user_input})

        messages = []
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"Model: {response}")
        history.append({"role": "assistant", "content": response})


def main():
    parser = argparse.ArgumentParser(description="Chat with fine-tuned Qwen model locally (CPU by default).")
    parser.add_argument(
        "--token",
        help="Token",
    )
    parser.add_argument(
        "--model",
        default="john-otis/Llama-3.2-1B-Instruct-LeBron",
        help="Hugging Face model repo id",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on; defaults to CPU",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.device, args.token)
    chat_loop(model, tokenizer, args.device)


if __name__ == "__main__":
    main()
