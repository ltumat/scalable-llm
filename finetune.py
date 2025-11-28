from typing import Any

from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only


@dataclass
class FinetuneConfig:
    model_name: str
    rank: int
    lora_alpha: int
    lora_dropout: float
    max_seq_length: int
    bias: str = "none"
    dtype: Any = None
    load_in_4bit: bool = True
    use_rank_stabilized_lora: bool = False
    seed: int = 42
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 60
    learning_rate: float = 2e-4
    fp16: bool = field(default_factory=lambda: not is_bfloat16_supported())
    bf16: bool = field(default_factory=is_bfloat16_supported)
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    output_dir: str = "outputs"
    report_to: str = "none"


def get_train_data() -> Dataset:
    return load_dataset("mlabonne/FineTome-100k", split="train")


def get_model_and_tokenizer(config: FinetuneConfig) -> tuple[FastLanguageModel, Any]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model=model,
        r=config.rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        use_rslora=config.use_rank_stabilized_lora,
        loftq_config=None,
    )

    return model, tokenizer


def finetune_model(config: FinetuneConfig, model: FastLanguageModel, tokenizer: Any, dataset: Dataset) -> None:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc=2,
        packing = False,
        args = SFTConfig(
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            fp16=config.fp16,
            bf16=config.bf16,
            logging_steps=config.logging_steps,
            optim=config.optim,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            seed=config.seed,
            output_dir=config.output_dir,
            report_to=config.report_to,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            # metric_for_best_model = "eval_loss",
        ),
    )

    # trainer = train_on_responses_only(
    #     trainer,
    #     instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    #     response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    # )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience = 10,
        early_stopping_threshold = 0.0,
    )

    trainer.add_callback(early_stopping_callback)

    # not sure we need early stopping with max_steps, but just in case

    trainer.train(resume_from_checkpoint=True)
    

if __name__ == "__main__":
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
        output_dir="outputs",
        report_to="none",
    )

    dataset = get_train_data()

    model, tokenizer = get_model_and_tokenizer(model_config)
    finetune_model(config=model_config, model=model, tokenizer=tokenizer, dataset=dataset)
