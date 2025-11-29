from pathlib import Path
import sys

import modal
from modal import FilePatternMatcher


APP_DIR = Path("/app")
if APP_DIR.exists():
    sys.path.insert(0, str(APP_DIR))

app = modal.App(name="finetune-qwen")

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
    .add_local_dir(".", "/app", ignore=IGNORE_PATTERNS)
)


@app.function(
    image=image,
    gpu="A100:2",
    timeout=60 * 60 * 1,
    volumes={"/outputs": vol},
    env={
        "NCCL_DEBUG": "WARN",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def main():
    """
    Launch multi-GPU training via torchrun. Each rank executes finetune.py,
    and Hugging Face Trainer handles distributed init.
    """
    import os
    import subprocess
    import torch

    nproc = max(1, torch.cuda.device_count())
    if nproc == 1:
        subprocess.run(["python", "finetune.py"], check=True)
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    script_path = "/app/finetune.py"
    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(nproc),
        "--master_addr",
        os.environ["MASTER_ADDR"],
        "--master_port",
        os.environ["MASTER_PORT"],
        script_path,
    ]
    subprocess.run(cmd, check=True)


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

    checkpoint_path = Path("/outputs/checkpoint-26100")
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
