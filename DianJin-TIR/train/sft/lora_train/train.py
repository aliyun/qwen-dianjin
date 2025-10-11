import os
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, load_from_disk
import transformers
from transformers import TrainingArguments, AutoTokenizer, Trainer, AutoModel
from datasets import load_dataset
import torch
import time
from peft import LoraConfig, get_peft_model
# from deepspeed.utils import logger
import logging
import deepspeed
import argparse
import yaml

# ------- Inference Config ------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='sft_config.yaml', help='Path to the config file')
args = parser.parse_args()
if os.path.exists(args.config):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        for key, value in config.items():
            print(key, value)
            setattr(args, key, value)
# ------- Inference Config ------------


if __name__ == "__main__":
    model_path = args.model_path
    dataset_path = args.train_data_save_folder
    max_length = args.train_max_tokens     

    config = LoraConfig(
                r=4,
                lora_alpha=16,
                target_modules=['gate_proj', 'k_proj', 'up_proj', 'down_proj', 'v_proj', 'o_proj', 'q_proj'],
                inference_mode=False,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
    model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            # device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model and tokenizer loaded successfully.")
    dataset = load_from_disk(dataset_path)

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, model=model, return_tensors="pt", padding=True, pad_to_multiple_of=8, max_length=max_length
    )

    training_args = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_safetensors=True,
        report_to="swanlab",
        output_dir=args.model_save_folder,
        do_train=True,
        max_length=max_length,
        packing=False,
        remove_unused_columns=True,
        deepspeed='ds_zero3_offload.json',
        dataloader_pin_memory=False,
        disable_tqdm=False,
        gradient_checkpointing=True,
        save_only_model=True
    )

    # 创建SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        peft_config=config
        # tokenizer=tokenizer,  # 添加tokenizer
    )
    # 开始训练
    engine = getattr(trainer.model, "ds_engine", None)
    if engine is None and hasattr(trainer.model, "module"):
        engine = getattr(trainer.model.module, "ds_engine", None)
    if engine:
        print("ZeRO stage:", getattr(engine, "zero_optimization_stage", "unknown"))
    trainer.train()