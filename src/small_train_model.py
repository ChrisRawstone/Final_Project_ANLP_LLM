#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_training.py

A script to test the data preprocessing and training loop on a small subset of the dataset.
This helps in diagnosing issues like `nan` losses during training.

Usage:
    python test_training.py

Ensure that the required datasets are available at the specified paths.
"""

# ------------------------------
# 1. Imports and Configuration
# ------------------------------

import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from datasets import load_from_disk
from tqdm import tqdm
import random
import numpy as np

# ------------------------------
# 2. Setup and Configuration
# ------------------------------

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Check for GPU availability
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Configuration parameters
model_name = "Qwen/Qwen2.5-0.5B"
train_path = "data/raw/eli5_qa_danish/train"          # Update this path if necessary
validation_path = "data/raw/eli5_qa_danish/validation"  # Update this path if necessary
output_dir = "./qwen2.5-0.5B-danish-pytorch-test"
batch_size = 2
num_epochs = 1  # Only one epoch for testing
learning_rate = 1e-5  # Lowered learning rate
weight_decay = 0.01
max_length = 512  # Maximum token length
gradient_accumulation_steps = 1  # No accumulation for testing
fp16 = False  # Disable mixed precision for testing
max_grad_norm = 1.0  # Gradient clipping

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# 3. Load Tokenizer and Model
# ------------------------------

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# **Add special tokens**
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

model.to(device)

# Ensure the tokenizer uses the special tokens
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------------
# 4. Load and Inspect Data
# ------------------------------

# Load the datasets from disk
print("Loading datasets...")
train_dataset = load_from_disk(train_path)
validation_dataset = load_from_disk(validation_path)

# Print dataset information
print("\ntrain_dataset: \n", train_dataset)
print("\nvalidation_dataset:\n", validation_dataset)

# Select a small subset for testing (e.g., first 1000 examples)
test_subset_size = 1000
small_train_dataset = train_dataset.select(range(test_subset_size))
print(f"\nSelected first {test_subset_size} examples from the training dataset for testing.")

# ------------------------------
# 5. Preprocessing Function
# ------------------------------

def preprocess_function(examples):
    """
    Preprocesses the dataset by constructing prompts, tokenizing, and aligning labels.
    Only the assistant's response is used as labels; the rest are ignored (-100).
    """
    inputs = []
    labels = []
    assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    missing_assistant_token = 0

    for query, passage in zip(examples['query'], examples['passage']):
        # Construct the prompt
        prompt = f"<|user|>{query}<|end_of_turn|><|assistant|>{passage}<|end_of_turn|>"

        # Tokenize the prompt
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True
        )
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        # Initialize labels with -100 to ignore in loss computation
        labels_ids = [-100] * len(input_ids)

        # Find the position of the assistant token
        try:
            assistant_token_position = input_ids.index(assistant_token_id)
            # Set labels for the assistant's response
            labels_ids[assistant_token_position + 1:] = input_ids[assistant_token_position + 1:]
        except ValueError:
            # Assistant token not found
            missing_assistant_token += 1
            # All labels remain -100, meaning this example will be ignored in loss computation

        inputs.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })
        labels.append(labels_ids)

    # Log the number of examples where the assistant token was not found
    if missing_assistant_token > 0:
        print(f"Number of examples where assistant token was not found: {missing_assistant_token}")

    return {
        'input_ids': [x['input_ids'] for x in inputs],
        'attention_mask': [x['attention_mask'] for x in inputs],
        'labels': labels
    }

# ------------------------------
# 6. Apply Preprocessing
# ------------------------------

print("\nPreprocessing the small training dataset...")
tokenized_train_dataset = small_train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=small_train_dataset.column_names
)

# Check how many examples have valid labels
valid_examples = [label for label in tokenized_train_dataset['labels'] if any(l != -100 for l in label)]
print(f"Number of valid examples with labels: {len(valid_examples)} out of {test_subset_size}")

# Optionally, filter out examples where labels are all -100
def filter_empty_labels(example):
    return any(label != -100 for label in example['labels'])

tokenized_train_dataset = tokenized_train_dataset.filter(filter_empty_labels)
print(f"After filtering, {len(tokenized_train_dataset)} examples remain.")

# ------------------------------
# 7. Create DataLoader
# ------------------------------

def collate_fn(batch):
    """
    Custom collate function to handle dynamic padding for variable-length sequences.
    """
    input_ids = [torch.tensor(example['input_ids']) for example in batch]
    attention_masks = [torch.tensor(example['attention_mask']) for example in batch]
    labels = [torch.tensor(example['labels']) for example in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    return {
        'input_ids': input_ids_padded.to(device),
        'attention_mask': attention_masks_padded.to(device),
        'labels': labels_padded.to(device),
    }

# Create DataLoader for the small subset
train_loader = DataLoader(
    tokenized_train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

print(f"\nCreated DataLoader with batch size {batch_size}.")

# ------------------------------
# 8. Initialize Optimizer and Scheduler
# ------------------------------

# Calculate total training steps
total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps

# Initialize the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Initialize the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,  # 10% of total steps for warm-up
    num_training_steps=total_steps
)

# Initialize GradScaler if using mixed precision
if fp16:
    scaler = torch.amp.GradScaler(enabled=fp16)
else:
    scaler = None

# ------------------------------
# 9. Define Training Steps
# ------------------------------

def run_training_steps(model, loader, optimizer, scheduler, scaler, device, num_steps=10):
    """
    Runs a specified number of training steps to test the training loop.
    """
    model.train()
    print("\nStarting training steps...")

    for step, batch in enumerate(tqdm(loader, desc="Training Steps")):
        if step >= num_steps:
            break

        # Forward pass
        if fp16:
            with torch.cuda.amp.autocast(device_type='cuda', enabled=fp16):
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss
        else:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss

        # Check if loss is nan
        if torch.isnan(loss):
            print(f"Step {step}: Loss is nan. Exiting training.")
            return

        # Backward pass
        if fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient clipping
        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        if fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Scheduler step
        scheduler.step()

        # Zero gradients
        optimizer.zero_grad()

        # Log loss
        print(f"Step {step + 1}: Loss = {loss.item():.4f}")

    print("Training steps completed successfully without encountering nan losses.")

# ------------------------------
# 10. Run the Test
# ------------------------------

if __name__ == "__main__":
    # Run a limited number of training steps to test
    run_training_steps(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        num_steps=10  # Number of steps to run for testing
    )

    print("\nTest script completed.")
