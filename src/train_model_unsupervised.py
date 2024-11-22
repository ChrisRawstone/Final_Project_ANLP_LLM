# ------------------------------
# 1. Imports and Configuration
# ------------------------------

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import DataLoader
from typing import Optional, List
from utils_unsupervised import (
    set_seed,
    run_training_steps,
)
import wandb

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_MODE"] = "online"

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
    data_files = {"train": "hf://datasets/alexandrainst/lexdk-open/data/train-00000-of-00001-693e1753187fc348.parquet"}
    batch_size = 4
    num_epochs = 1  # Adjust as needed
    learning_rate = 5e-10
    weight_decay = 0.01
    block_size = 500  # Adjust based on your data
    gradient_accumulation_steps = 4
    fp16 = True  # Enable mixed precision
    max_grad_norm = 1.0  # Gradient clipping
    num_workers = 4  # DataLoader workers
    output_dir = "models/unsupervised_finetuning"  # Directory to save models

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize wandb
    wandb.login(key="d8f470718404594a18af450a403047ce84172434")
    wandb.init(
        project="danish-unsupervised-finetuning",
        config={
            "model_name": model_name,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "block_size": block_size,
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Print trainable parameters
    print(f"Trainable parameters: {model.num_parameters()}")

    model.to(device)

    # Ensure the tokenizer uses the pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------
    # 4. Load and Preprocess Data
    # ------------------------------

    # Load the dataset from the Parquet file
    from datasets import load_dataset

    dataset = load_dataset('parquet', data_files=data_files, split='train')

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=dataset.column_names,
    )

    # Group texts into blocks
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples["input_ids"])
        # Drop the small remainder
        total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_workers,
    )

    # ------------------------------
    # 5. Create DataLoader
    # ------------------------------

    # Initialize Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    train_loader = DataLoader(
        lm_datasets,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
    )

    # ------------------------------
    # 6. Initialize Optimizer and Scheduler
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
    # 7. Prepare Evaluation Prompts
    # ------------------------------

    # Example Danish prompts for evaluation
    evaluation_prompts = [
        "Hvordan laver jeg en kop kaffe?",
        "Hvad er meningen med livet?",
        "Kan du forklare kvantemekanik?",
        # Add more prompts as needed
    ]

    # ------------------------------
    # 8. Run the Training
    # ------------------------------

    # Run the training with the training loop
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
