#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_model.py

A script to train a causal language model on Danish question-answering data.
Includes data preprocessing, training loop with evaluation, learning rate scheduling, and logging with wandb.

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

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

wandb.login(key='aa3e3c46cc563c788908b8b2be3dcfb1ee6f109c')

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
    model_name = "Qwen/Qwen2.5-0.5B"
    train_path = "data/raw/eli5_qa_danish/train"          # Update this path if necessary
    validation_path = "data/raw/eli5_qa_danish/validation" # Update this path if necessary
    batch_size = 2
    num_epochs = 3  # Adjust as needed
    learning_rate = 1e-5
    weight_decay = 0.01
    max_length = 256  # Adjust based on your data
    gradient_accumulation_steps = 4
    fp16 = True  # Enable mixed precision
    max_grad_norm = 1.0  # Gradient clipping
    num_workers = 4  # DataLoader workers

    # Initialize wandb
    wandb.init(
        project="danish-small-LLM",
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
        }
    )

    # ------------------------------
    # 3. Load Tokenizer and Model
    # ------------------------------

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add special tokens
    special_tokens_dict = {
        'additional_special_tokens': ['<|user|>', '<|assistant|>', '<|end_of_turn|>']
    }
    num_added_toks: int = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens to the tokenizer.")

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Resize model embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to {len(tokenizer)} tokens.")

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

    # Select a small subset for testing (e.g., first 1000 examples)
    test_subset_size: int = 1000
    small_train_dataset = train_dataset.select(range(test_subset_size))
    print(f"\nSelected first {test_subset_size} examples from the training dataset for testing.")

    # ------------------------------
    # 5. Apply Preprocessing
    # ------------------------------

    print("\nPreprocessing the small training dataset...")
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

    # Create DataLoader for the small subset
    train_loader: DataLoader = create_dataloader(
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
    scaler = torch.amp.GradScaler(device_type="cuda", enabled=fp16) if fp16 else None

    # ------------------------------
    # 8. Prepare Evaluation Prompts
    # ------------------------------

    # Example Danish question prompts
    evaluation_prompts: = [
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
        num_steps_per_epoch=None  # Set to limit steps per epoch if needed
    )

    print("\nTraining script completed.")

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
