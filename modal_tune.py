from pathlib import Path
import sys
import modal
from modal import FilePatternMatcher

APP_DIR = Path("/app")
if APP_DIR.exists():
    sys.path.insert(0, str(APP_DIR))

app = modal.App(name="finetune-llama")

vol = modal.Volume.from_name("model-checkpoints", create_if_missing=True)
IGNORE_PATTERNS = FilePatternMatcher(
    ".venv/**",
    ".git/**",
    "**/__pycache__/**",
    "model-checkpoints/**",
    ".modal_cache/**",
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install(
        # bitsandbytes and triton only publish Linux/Windows wheels; pull them inside the Modal image.
        "bitsandbytes==0.43.2",
        # xformers 0.0.24 pairs with torch 2.2.x; use matching triton.
        "triton==2.2.0",
        "xformers==0.0.24",
        # Torch 2.2.2 was compiled against NumPy 1.x headers; keep NumPy <2 to avoid runtime crashes.
        "numpy<2",
    )
    .add_local_dir(".", "/app", ignore=IGNORE_PATTERNS)
)


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 2,
    volumes={"/outputs": vol},
)
def main():
    from finetune import (
        FinetuneConfig,
        get_model_and_tokenizer,
        finetune_model,
        get_train_data,
    )
    from unsloth import is_bfloat16_supported

    model_config = FinetuneConfig(
        model_name="unsloth/Llama-3.2-1B-bnb-4bit",
        rank=16,
        lora_alpha=16,
        lora_dropout=0.0,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        output_dir="/outputs",
        report_to="none",
    )

    dataset = get_train_data()

    model, tokenizer = get_model_and_tokenizer(model_config)
    finetune_model(config=model_config, model=model, tokenizer=tokenizer, dataset=dataset)


@app.local_entrypoint()
def run():
    main.remote()
