#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_model.py

A script to train a causal language model on Danish question-answering data.
Includes data preprocessing, training loop with evaluation, learning rate scheduling, and model checkpointing.

Usage:
    python train_model.py

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
    get_cosine_schedule_with_warmup,  # Optional: For alternative scheduler
)
from datasets import load_from_disk
from tqdm import tqdm
import random
import numpy as np

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
train_path = "data/raw/eli5_qa_danish/train"             # Update this path if necessary
validation_path = "data/raw/eli5_qa_danish/validation" # Update this path if necessary
output_dir = "./models/initial_test/"
batch_size = 2
num_epochs = 3  # Adjust as needed
learning_rate = 1e-5
weight_decay = 0.01
max_length = 256  # Adjust based on your data
gradient_accumulation_steps = 4
fp16 = True  # Enable mixed precision
max_grad_norm = 1.0  # Gradient clipping
num_workers = 4  # DataLoader workers

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

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
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'labels': labels_padded,
    }

# Create DataLoader for the small subset
train_loader = DataLoader(
    tokenized_train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers
)

print(f"\nCreated DataLoader with batch size {batch_size} and {num_workers} workers.")

# ------------------------------
# 8. Initialize Optimizer and Scheduler
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

# Alternatively, use cosine scheduler
# scheduler = get_cosine_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=total_steps // 10,
#     num_training_steps=total_steps
# )

# Initialize GradScaler if using mixed precision
if fp16:
    scaler = torch.amp.GradScaler(enabled=fp16)
else:
    scaler = None

# ------------------------------
# 9. Define Evaluation Function
# ------------------------------

def evaluate(model, tokenizer, device, prompts, max_new_tokens=256):
    """
    Generates responses for a list of prompts using sampling methods.
    """
    model.eval()
    responses = []
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                temperature=0.8,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            # Decode only the generated tokens
            generated_tokens = output_ids[0][input_ids.shape[-1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response.strip())
    model.train()
    return responses

# ------------------------------
# 10. Prepare Evaluation Prompts
# ------------------------------

# Example Danish question prompts
evaluation_prompts = [
    "<|user|>Hvordan laver jeg en kop kaffe?<|end_of_turn|>",
    "<|user|>Hvad er meningen med livet?<|end_of_turn|>",
    "<|user|>Kan du forklare kvantemekanik?<|end_of_turn|>",
    # Add more prompts as needed
]

# ------------------------------
# 11. Define Training Function with Evaluation and Checkpointing
# ------------------------------

def run_training_steps(model, loader, optimizer, scheduler, scaler, device, num_epochs=3, num_steps_per_epoch=None):
    """
    Runs the training loop with evaluation and checkpointing.
    """
    model.train()
    print("\nStarting training...")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}")):
            if num_steps_per_epoch and step >= num_steps_per_epoch:
                break

            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with autocast
            with torch.autocast(device_type=device.type, enabled=fp16):
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps  # Normalize loss

            # Check if loss is nan
            if torch.isnan(loss):
                print(f"Epoch {epoch + 1}, Step {step + 1}: Loss is nan. Exiting training.")
                return

            # Backward pass
            if fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate loss
            epoch_loss += loss.item() * gradient_accumulation_steps  # Multiply back to original loss

            # Optimizer step with gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16:
                    # Unscale gradients and clip
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Log loss every 10 steps
            if (step + 1) % 10 == 0:
                avg_loss = epoch_loss / ((step + 1) / gradient_accumulation_steps)
                print(f"Epoch {epoch + 1}, Step {step + 1}: Avg Loss = {avg_loss:.4f}")

        # Handle remaining gradients if steps are not divisible by gradient_accumulation_steps
        if (step + 1) % gradient_accumulation_steps != 0:
            if fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # After each epoch, evaluate the model
        print("\nEvaluating the model with Danish prompts...")
        responses = evaluate(model, tokenizer, device, evaluation_prompts)
        for prompt, response in zip(evaluation_prompts, responses):
            print(f"\nPrompt: {prompt}\nResponse: {response}")

        # Save a checkpoint after each epoch
        checkpoint_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch + 1}")
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"\nSaved model checkpoint to {checkpoint_path}")

    # Save the final model
    final_model_path = os.path.join(output_dir, "final-model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\nSaved final model to {final_model_path}")

    print("\nTraining completed successfully.")

# ------------------------------
# 12. Run the Training
# ------------------------------

if __name__ == "__main__":
    # Run the training with the enhanced training loop
    run_training_steps(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        num_epochs=num_epochs,  # Number of epochs
        num_steps_per_epoch=None  # Set to limit steps per epoch if needed
    )

    print("\nTraining script completed.")
