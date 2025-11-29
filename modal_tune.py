from pathlib import Path
import sys

import modal
from modal import FilePatternMatcher


APP_DIR = Path("/app")
if APP_DIR.exists():
    sys.path.insert(0, str(APP_DIR))

app = modal.App(name="finetune-qwen")

vol = modal.Volume.from_name("model-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir(
        ".", 
        "/app", 
        ignore=[
            ".venv", 
            ".git", 
            "__pycache__", 
            "model-checkpoints", 
            ".modal_cache",
            "**/.venv/**",
            "**/.git/**",
            "**/__pycache__/**"
        ]
    )
)


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 2,
    volumes={"/outputs": vol},
)
def main():
    from finetune import FinetuneConfig, finetune_model

    config = FinetuneConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        dataset_name="lebron_james/lebron_interviews_cleaned.jsonl",
        dataset_split="train",
        max_seq_length=1024,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=5,
        warmup_steps=50,
        max_steps=1000000,
        learning_rate=2e-4,
        logging_steps=5,
        save_steps=50,
        output_dir="/outputs",
    )

    finetune_model(config)


@app.local_entrypoint()
def run():
    main.remote()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/outputs": vol},
)
def push_to_hub():
    import os
    from huggingface_hub import HfApi

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    user = api.whoami(token=token)["name"]
    repo_id = f"{user}/Qwen3-4B-Instruct-FineTome"
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    checkpoint_path = Path("/outputs/checkpoint-200")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(checkpoint_path),
        path_in_repo=".",
        commit_message="Upload fine-tuned weights",
    )
    return repo_id


@app.local_entrypoint()
def push():
    repo_id = push_to_hub.remote()
    print(f"Uploaded to https://huggingface.co/{repo_id}")
