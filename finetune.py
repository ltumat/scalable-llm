import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer


def _bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


@dataclass
class FinetuneConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    dataset_name: str = "mlabonne/FineTome-100k"
    dataset_split: str = "train"
    max_seq_length: int = 1024
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 50
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_train_epochs: int = 1
    max_steps: int = 200
    logging_steps: int = 5
    save_steps: int = 100
    output_dir: str = "outputs"
    seed: int = 42
    gradient_checkpointing: bool = False
    gradient_checkpointing_kwargs: dict | None = field(default_factory=lambda: {"use_reentrant": False})
    ddp_find_unused_parameters: bool | None = False
    torch_dtype: Any = field(default_factory=lambda: torch.bfloat16 if _bf16_supported() else torch.float16)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


def get_train_data(config: FinetuneConfig) -> Dataset:
    ds = load_dataset(config.dataset_name, split=config.dataset_split)

    def build_text(example):
        conv = example.get("conversations", [])
        parts = []
        for turn in conv:
            content = turn.get("value")
            if not content:
                continue
            role = turn.get("from", "").lower()
            prefix = "User" if role in ("human", "user") else "Assistant"
            parts.append(f"{prefix}: {content}")
        example["text"] = "\n\n".join(parts)
        return example

    ds = ds.map(build_text, remove_columns=["conversations"])  # keep original cols
    ds = ds.filter(lambda x: isinstance(x.get("text"), str) and len(x["text"].strip()) > 0)
    return ds


def get_tokenizer(config: FinetuneConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=False,
        trust_remote_code=True,
        token = os.environ["HF_LLAMA_3_2"],
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = config.max_seq_length
    return tokenizer


def get_model(config: FinetuneConfig):
    return AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=config.torch_dtype,
        device_map=None,
        trust_remote_code=True,
        token = os.environ["HF_LLAMA_3_2"],
    )


def build_trainer(config: FinetuneConfig, model, tokenizer, dataset: Dataset) -> SFTTrainer:
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_config = SFTConfig(
        dataset_text_field="text",
        max_length=config.max_seq_length,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        output_dir=config.output_dir,
        bf16=config.torch_dtype == torch.bfloat16,
        fp16=config.torch_dtype == torch.float16,
        seed=config.seed,
        packing=False,
        report_to="none",
        dataset_num_proc=2,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs=config.gradient_checkpointing_kwargs,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
    )

    return SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )


def finetune_model(config: FinetuneConfig) -> None:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    dataset = get_train_data(config)

    tokenizer = get_tokenizer(config)
    model = get_model(config)

    trainer = build_trainer(config, model, tokenizer, dataset)
    trainer.train()

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    config = FinetuneConfig(
        #model_name="Qwen/Qwen3-4B-Instruct-2507",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        dataset_name="mlabonne/FineTome-100k",
        dataset_split="train",
        max_seq_length=1024,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        warmup_steps=50,
        max_steps=100000,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        output_dir="/outputs",
    )

    finetune_model(config)
