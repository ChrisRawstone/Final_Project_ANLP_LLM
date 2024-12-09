#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_model_lora.py

A script to train a causal language model on Danish question-answering data.
Includes data preprocessing, training loop with evaluation, learning rate scheduling,
logging with wandb, and saving the model after each epoch.

Usage:
    python train_model_lora.py

Ensure that the required datasets are available at the specified paths.
"""

# ------------------------------
# 1. Imports and Configuration
# ------------------------------

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)
from torch.utils.data import DataLoader
from typing import Optional, List
from src.utils import (
    set_seed,
    load_datasets,
    preprocess_function,
    filter_empty_labels,
    create_dataloader,
    evaluate,
    run_training_steps,
)
import wandb

import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import PEFT modules for LoRA and LoftQ
from peft import LoraConfig, get_peft_model, replace_lora_weights_loftq

# Ensure you are logged into wandb
wandb.login()

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Disable wandb if not needed
# os.environ["WANDB_MODE"] = "disabled"

# ------------------------------
# 2. Setup and Configuration
# ------------------------------

def main() -> None:
    # Set random seeds for reproducibility
    seed = 42
    set_seed(seed)

    # Check for GPU availability
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Configuration parameters
    model_name = "Qwen/Qwen2.5-0.5B"
    train_path = "data/processed/instruct_train_dataset"
    validation_path = "data/processed/instruct_val_dataset"
    batch_size = 2
    num_epochs = 1  # Adjust as needed
    learning_rate = 5e-5
    weight_decay = 0.01
    max_length = 256  # Adjust based on your data
    gradient_accumulation_steps = 4
    fp16 = True  # Enable mixed precision
    max_grad_norm = 1.0  # Gradient clipping
    num_workers = 4  # DataLoader workers
    output_dir = "models/lora_first_iteration"  # Directory to save models
    save_steps = 500  # Adjust save steps as needed

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize wandb
    wandb.init(
        project="danish-qa-model",
        config={
            "model_name": model_name,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_length": max_length,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "fp16": fp16,
            "max_grad_norm": max_grad_norm,
            "num_workers": num_workers,
            "seed": seed,
            "output_dir": output_dir,
        },
        reinit=True,
    )

    # ------------------------------
    # 3. Load Tokenizer and Model with LoRA and Quantization
    # ------------------------------

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add special tokens
    special_tokens_dict = {
        "additional_special_tokens": ["<|user|>", "<|assistant|>", "<|end_of_turn|>"]
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added_toks} special tokens to the tokenizer.")

    # Define the quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load the model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Resize model embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Resized model embeddings to {len(tokenizer)} tokens.")



    # Create LoRA configuration with correct target modules
    lora_config = LoraConfig(
        r=16,  # Low-rank update matrices rank
        lora_alpha=32,  # Scaling factor
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Adjusted module names
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Wrap the model with LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters for verification
    trainable_params, all_param = model.get_nb_trainable_parameters()

    logger.info(
        f"Trainable params: {trainable_params:,d} || "
        f"All params: {all_param:,d} || "
        f"Trainable%: {100 * trainable_params / all_param:.4f}%"
    )


    # Ensure the tokenizer uses the special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------
    # 4. Load and Inspect Data
    # ------------------------------

    # Load the datasets from disk
    logger.info("Loading datasets...")
    train_dataset, validation_dataset = load_datasets(train_path, validation_path)

    # Log dataset information
    logger.info(f"Loaded {len(train_dataset)} training examples.")
    logger.info(f"Loaded {len(validation_dataset)} validation examples.")

    # No need to select a subset; use the entire dataset
    small_train_dataset = train_dataset.shuffle(seed=seed).select(range(1000))

    # ------------------------------
    # 5. Apply Preprocessing
    # ------------------------------

    logger.info("Preprocessing the training dataset...")
    tokenized_train_dataset = small_train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=small_train_dataset.column_names,
    )

    # Filter out examples where labels are all -100
    tokenized_train_dataset = tokenized_train_dataset.filter(filter_empty_labels)
    logger.info(f"After filtering, {len(tokenized_train_dataset)} examples remain.")

    # Preprocess the validation dataset
    logger.info("Preprocessing the validation dataset...")
    tokenized_val_dataset = validation_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=validation_dataset.column_names,
    )

    tokenized_val_dataset = tokenized_val_dataset.filter(filter_empty_labels)
    logger.info(f"After filtering, {len(tokenized_val_dataset)} validation examples remain.")

    # ------------------------------
    # 6. Create DataLoaders
    # ------------------------------

    # Create DataLoaders for training and validation datasets
    train_loader = create_dataloader(
        tokenized_train_dataset,
        tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    val_loader = create_dataloader(
        tokenized_val_dataset,
        tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    logger.info(
        f"Created DataLoaders with batch size {batch_size} and {num_workers} workers."
    )

    # ------------------------------
    # 7. Initialize Optimizer and Scheduler
    # ------------------------------

    # Calculate total training steps
    total_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs
    logger.info(f"Total training steps: {total_steps}")

    # Only include trainable parameters in the optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Log optimizer's parameters
    num_optimizer_params = sum(p.numel() for p in optimizer.param_groups[0]['params'])
    logger.info(f"Number of parameters in optimizer: {num_optimizer_params}")

    # Initialize the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,  # 10% of total steps for warm-up
        num_training_steps=total_steps,
    )

    # Initialize GradScaler if using mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=fp16) if fp16 else None

    # ------------------------------
    # 8. Prepare Evaluation Prompts
    # ------------------------------

    # Example Danish question prompts
    evaluation_prompts = [
        "<|user|>Hvordan laver jeg en kop kaffe?<|end_of_turn|>",
        "<|user|>Hvad er meningen med livet?<|end_of_turn|>",
        "<|user|>Kan du forklare kvantemekanik?<|end_of_turn|>",
        # Add more prompts as needed
    ]

    # ------------------------------
    # 9. Run the Training
    # ------------------------------

    # Run the training with the enhanced training loop
    run_training_steps(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        evaluation_prompts=evaluation_prompts,
        tokenizer=tokenizer,
        num_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        max_grad_norm=max_grad_norm,
        num_steps_per_epoch=None,
        output_dir=output_dir,
        save_steps=save_steps,
    )

    logger.info("Training script completed.")

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()