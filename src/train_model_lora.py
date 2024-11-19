#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_model.py

A script to train a causal language model on Danish question-answering data.
Includes data preprocessing, training loop with evaluation, learning rate scheduling,
logging with wandb, and saving the model after each epoch.

Usage:
    python train_model.py

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
)
from torch.utils.data import DataLoader
from typing import Optional, List
from utils import (
    set_seed,
    load_datasets,
    preprocess_function,
    filter_empty_labels,
    create_dataloader,
    evaluate,
    run_training_steps,
)
import wandb  # Import wandb

# Import PEFT modules for LoRA
from peft import LoraConfig, get_peft_model

# Ensure you are logged into wandb
wandb.login()

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------------
# 2. Setup and Configuration
# ------------------------------

def main() -> None:
    # Set random seeds for reproducibility
    seed = 42
    set_seed(seed)

    # Check for GPU availability
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Configuration parameters
    model_name = "Qwen/Qwen2.5-1.5B"
    train_path = "data/raw/eli5_qa_danish/train"             # Update this path if necessary
    validation_path = "data/raw/eli5_qa_danish/validation"   # Update this path if necessary
    batch_size = 16
    num_epochs = 1  # Adjust as needed
    learning_rate = 5e-5
    weight_decay = 0.01
    max_length = 256  # Adjust based on your data
    gradient_accumulation_steps = 4
    fp16 = True  # Enable mixed precision
    max_grad_norm = 1.0  # Gradient clipping
    num_workers = 4  # DataLoader workers
    output_dir = "models/long_train_hpc"  # Directory to save models

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
        reinit=True
    )

    # ------------------------------
    # 3. Load Tokenizer and Model with LoRA
    # ------------------------------

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add special tokens
    special_tokens_dict = {
        'additional_special_tokens': ['<|user|>', '<|assistant|>', '<|end_of_turn|>']
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens to the tokenizer.")

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Resize model embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to {len(tokenizer)} tokens.")

    # Create LoRA configuration
    lora_config = LoraConfig(
        r=16,  # Low-rank update matrices rank
        lora_alpha=32,  # Scaling factor
        target_modules=["W_pack", "o_proj"],  # Target modules in Qwen model
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Wrap the model with LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters for verification
    model.print_trainable_parameters()

    # Move model to device
    model.to(device)

    # Ensure the tokenizer uses the special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------
    # 4. Load and Inspect Data
    # ------------------------------

    # Load the datasets from disk
    print("Loading datasets...")
    train_dataset, validation_dataset = load_datasets(train_path, validation_path)

    # Print dataset information
    print("\ntrain_dataset: \n", train_dataset)
    print("\nvalidation_dataset:\n", validation_dataset)

    # Use the full training dataset
    small_train_dataset = train_dataset

    # ------------------------------
    # 5. Apply Preprocessing
    # ------------------------------

    print("\nPreprocessing the training dataset...")
    tokenized_train_dataset = small_train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=small_train_dataset.column_names
    )

    # Filter out examples where labels are all -100
    tokenized_train_dataset = tokenized_train_dataset.filter(filter_empty_labels)
    print(f"After filtering, {len(tokenized_train_dataset)} examples remain.")

    # ------------------------------
    # 6. Create DataLoader
    # ------------------------------

    # Create DataLoader for the training dataset
    train_loader = create_dataloader(
        tokenized_train_dataset,
        tokenizer,
        batch_size=batch_size,
        num_workers=num_workers
    )

    print(f"\nCreated DataLoader with batch size {batch_size} and {num_workers} workers.")

    # ------------------------------
    # 7. Initialize Optimizer and Scheduler
    # ------------------------------

    # Calculate total training steps
    total_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,  # 10% of total steps for warm-up
        num_training_steps=total_steps
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
        loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        evaluation_prompts=evaluation_prompts,
        tokenizer=tokenizer,
        num_epochs=num_epochs,  # Number of epochs
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        max_grad_norm=max_grad_norm,
        num_steps_per_epoch=None,  # Set to limit steps per epoch if needed
        output_dir=output_dir        # Pass the output directory for saving models
    )

    print("\nTraining script completed.")
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
