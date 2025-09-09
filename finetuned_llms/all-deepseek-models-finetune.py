"""
===============================================================================
 DeepSeek-R1 LoRA Fine-Tuning Script (8-bit Quantization)
===============================================================================

This script performs supervised fine-tuning of the quantized (8-bit) 
DeepSeek-R1-Distill-Llama-70B language model using the LoRA method via PEFT. 
It processes structured conversational datasets using Hugging Face's chat 
template formatting, prepares the model for efficient training, and saves the 
resulting LoRA adapter weights after training.

This script leverages 8-bit quantization to reduce memory usage and supports proxy-based and offline environments for flexible deployment.

Note: This script is intended for academic reference only.
"""

import pandas as pd
from datasets import Dataset, load_dataset

import torch

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTTrainer, SFTConfig
import bitsandbytes as bnb

from dotenv import load_dotenv

import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def setup_proxy():

    http_proxy = os.getenv("HTTP_PROXY", "")
    https_proxy = os.getenv("HTTPS_PROXY", "")

    if http_proxy:
        os.environ["HTTP_PROXY"] = http_proxy
        logger.info(f"HTTP_PROXY set to: {http_proxy}")

    if https_proxy:
        os.environ["HTTPS_PROXY"] = https_proxy
        logger.info(f"HTTPS_PROXY set to: {https_proxy}")
    
    if not http_proxy and not https_proxy:
        logger.info("Setting offline mode")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    else:
        logger.info("Proxy detected -> enabling online mode")

def main():
    
    logger.info("### Starting DeepSeek LoRA Fine-tuning ###")
    
    setup_proxy()
    
    set_seed(42)

    model_id = "/<my-path>/DeepSeek-R1-Distill-Llama-70B"
    
    # dataset_path = "samples/shap_finetune.jsonl"
    # dataset_path = "samples/gat_finetune.jsonl"
    dataset_path = "samples/combined_finetune.jsonl"
    
    # new_model_name = "deepseek-r1-70b-it-finetuned-xgb-8bit"
    # new_model_name = "deepseek-r1-70b-it-finetuned-gat-8bit"
    new_model_name = "deepseek-r1-70b-it-finetuned-hybrid-8bit"

    logger.info(f"Base Model: {model_id}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output LoRA Adapter: {new_model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=True,
    )

    logger.info("Loading base model with 8-Bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        logger.info("Set pad token to eos token")

    logger.info(f"Loading and preparing dataset from {dataset_path}...")
    dataset = load_dataset('json', data_files=dataset_path, split='train')
    original_size = len(dataset)
    logger.info(f"Original dataset size: {original_size} samples")
    
    dataset = dataset.shuffle(seed=42)
    
    def format_chat_template(row):

        messages = row.get("messages", [])
        if not messages: 
            return {"text": ""}
        
        system_prompt = ""
        conversation = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt += msg["content"] + "\n"
            else:
                conversation.append(msg)
        
        if system_prompt and conversation and conversation[0]["role"] == "user":
            conversation[0]["content"] = f"{system_prompt.strip()}\n\n{conversation[0]['content']}"
        
        formatted_messages = [msg for msg in conversation if msg["role"] in ["user", "assistant"]]
        
        if formatted_messages:
            row["text"] = tokenizer.apply_chat_template(
                formatted_messages, 
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            row["text"] = ""
            
        return row

    logger.info("Applying chat template formatting...")
    dataset = dataset.map(format_chat_template, desc="Formatting chat templates")
    dataset = dataset.filter(lambda example: len(example['text']) > 10)
    final_size = len(dataset)
    
    logger.info(f"Final dataset size: {final_size} samples (filtered {original_size - final_size} empty/short samples)")
    
    if final_size > 0:
        logger.info("Example of formatted sample:")
        logger.info(dataset[0]['text'][:500] + "..." if len(dataset[0]['text']) > 500 else dataset[0]['text'])
        
    logger.info("\nTokenizing dataset manually (truncation + padding)...")

    tokenized_dataset = dataset.map(
        lambda row: tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=2048
        ),
        batched=True,
        remove_columns=dataset.column_names
    )

    logger.info("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=16,                   
        lora_alpha=16,         
        lora_dropout=0.05,      
        bias="none",         
        task_type="CAUSAL_LM",  
        target_modules=[        
            "q_proj", "v_proj", "k_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"      
        ],
    )

    logger.info("Applying LoRA configuration...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        dataset_text_field=None,
        max_seq_length=2048,
        output_dir=new_model_name,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        optim="adamw_torch",
        num_train_epochs=1,
        logging_steps=25,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        report_to="none",
        seed=42,
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        packing=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False, 
    )

    logger.info("Setting up SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer, 
    )

    logger.info("\nStarting 8-Bit LoRA fine-tuning")
    logger.info(f"Training on {final_size} samples")
    logger.info(f"Target modules: {peft_config.target_modules}")
    logger.info(f"Will save to: ./{new_model_name}/")
    
    trainer.train()
    
    logger.info("\nFine-tuning completed")

    logger.info(f"\nSaving LoRA adapter to ./{new_model_name}")
    trainer.save_model()
    
    save_path = f"./{new_model_name}"
    if os.path.exists(save_path):
        files = os.listdir(save_path)
        logger.info(f"Files saved in {save_path}:")
        for file in sorted(files):
            file_path = os.path.join(save_path, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"{file} ({size_mb:.2f} MB)")
    
    logger.info("Fine-tuning completed successfully!")
    logger.info(f"LoRA adapter saved to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    main()