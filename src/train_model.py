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
import argparse
import os
import torch
from datetime import datetime
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
    run_training_steps,
)

from src.evaluation import evaluate_scandeval
from parser import get_args
import wandb  # Import wandb

# Replace with your actual wandb API key or ensure you are logged in via the command line
wandb.login(key="83fb1d160dc4cb3bbaceadab26cba368ebced6c6")

# Suppresss tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["WANDB_MODE"] = "Disabled"

# ------------------------------
# 2. Setup and Configuration
# ------------------------------

def main(args) -> None:

    # Configuration parameters
    model_name = "Qwen/Qwen2.5-0.5B"
    train_path = "data/processed/instruct_train_dataset"
    validation_path = "data/processed/instruct_val_dataset"
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    max_length = args.max_length
    gradient_accumulation_steps = args.gradient_accumulation_steps
    fp16 = args.fp16
    max_grad_norm = args.max_grad_norm
    num_workers = args.num_workers
    seed = args.seed

    # Set random seeds for reproducibility    
    set_seed(seed)

    # Check for GPU availability
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")


    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
       
    output_dir = f"models/instruction/{timestamp}"
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # save the configuration

    with open(f"{output_dir}/config.txt", "w") as f:
        f.write(f"model_name: {model_name}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"num_epochs: {num_epochs}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"weight_decay: {weight_decay}\n")
        f.write(f"max_length: {max_length}\n")
        f.write(f"gradient_accumulation_steps: {gradient_accumulation_steps}\n")
        f.write(f"fp16: {fp16}\n")
        f.write(f"max_grad_norm: {max_grad_norm}\n")
        f.write(f"num_workers: {num_workers}\n")
        f.write(f"seed: {seed}\n")

    # Initialize wandb
    wandb.init(
        project="danish-qa-model",
        name=f"instruction-{timestamp}",
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
    # 3. Load Tokenizer and Model
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

    # Print trainable parameters
    print(f"Trainable parameters: {model.num_parameters()}")

    # Resize model embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to {len(tokenizer)} tokens.")

    model.to(device)

    # Ensure the tokenizer uses the special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token_id

    # ------------------------------
    # 4. Load and Inspect Data
    # ------------------------------

    # Load the datasets from disk
    print("Loading datasets...")
    train_dataset, validation_dataset = load_datasets(train_path, validation_path)

    # Print dataset information
    print("\ntrain_dataset: \n", train_dataset)
    print("\nvalidation_dataset:\n", validation_dataset)

    # No need to select a subset; use the entire dataset # debug, remember to remove
    small_train_dataset = train_dataset    #.select(range(1000))
    validation_dataset = validation_dataset   #.select(range(250))
    # ------------------------------
    # 5. Apply Preprocessing
    # ------------------------------
      

    print("\nPreprocessing the training dataset...")
    tokenized_train_dataset = small_train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=small_train_dataset.column_names
    )

    print(f"Before filtering, {len(tokenized_train_dataset)} examples remain.")
    # Filter out examples where labels are all -100
    tokenized_train_dataset = tokenized_train_dataset.filter(filter_empty_labels)
    print(f"After filtering, {len(tokenized_train_dataset)} examples remain.")

    # Preprocess the validation dataset
    print("\nPreprocessing the validation dataset...")
    tokenized_val_dataset = validation_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=validation_dataset.column_names
    )

    tokenized_val_dataset = tokenized_val_dataset.filter(filter_empty_labels)
    print(f"After filtering, {len(tokenized_val_dataset)} validation examples remain.")

    # ------------------------------
    # 6. Create DataLoaders
    # ------------------------------

    # Create DataLoaders for training and validation datasets
    train_loader = create_dataloader(
        tokenized_train_dataset,
        tokenizer,
        batch_size=batch_size,
        num_workers=num_workers
    )

    val_loader = create_dataloader(
        tokenized_val_dataset,
        tokenizer,
        batch_size=batch_size,
        num_workers=num_workers
    )

    
    
    print(f"\nCreated DataLoaders with batch size {batch_size} and {num_workers} workers.")

    # ------------------------------
    # 7. Initialize Optimizer and Scheduler
    # ------------------------------

    # Calculate total training steps 
    # (steps where parameters are updated, this is the number of backward passes)
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
        train_loader=train_loader,
        val_loader=val_loader,  # Pass the validation DataLoader
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
        output_dir=output_dir
    )

    print("\nTraining script completed.")

    # Finish wandb run
    wandb.finish()

    evaluate_scandeval(MODEL_DIR=output_dir, RESULT_DIR=f"result/instruction/{timestamp}")



if __name__ == "__main__":
    args = get_args()
    main(args)