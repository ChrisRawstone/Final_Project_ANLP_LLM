# Import necessary libraries
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
import torch.nn.functional as F
import random
import numpy as np

# ------------------------------
# 1. Setup and Configuration
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
train_path = "data/raw/eli5_qa_danish/train"
validation_path = "data/raw/eli5_qa_danish/validation"
output_dir = "./qwen2.5-0.5B-danish-pytorch"
batch_size = 2
num_epochs = 3
learning_rate = 1e-5  # Lowered learning rate
weight_decay = 0.01
max_length = 512  # Maximum token length
save_steps = 10000
eval_steps = 5000
logging_steps = 500
gradient_accumulation_steps = 1  # Adjust based on GPU memory
fp16 = True  # Use mixed precision
save_total_limit = 2
max_grad_norm = 1.0  # Gradient clipping

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# 2. Data Loading and Preprocessing
# ------------------------------

# Load the datasets from disk
train_dataset = load_from_disk(train_path)
validation_dataset = load_from_disk(validation_path)

# Print dataset information
print("train_dataset: \n", train_dataset)
print("validation_dataset:\n", validation_dataset)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# **Add special tokens**
special_tokens_dict = {'additional_special_tokens': ['<|user|>', '<|assistant|>', '<|end_of_turn|>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Added {num_added_toks} special tokens.")

# Load the model and resize embeddings
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

model.to(device)

# Ensure the tokenizer uses the special tokens
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Preprocessing function to tokenize the data and create labels
def preprocess_function(examples):
    inputs = []
    labels = []
    assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    missing_assistant_token = 0
    for query, passage in zip(examples['query'], examples['passage']):
        # Construct the prompt
        prompt = f"<|user|>{query}<|end_of_turn|><|assistant|>{passage}<|end_of_turn|>"
        # Tokenize the prompt
        tokenized = tokenizer(prompt, truncation=True, max_length=max_length, padding=False)
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
            # You can decide how to handle this case
            labels_ids = [-100] * len(input_ids)

        inputs.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })
        labels.append(labels_ids)

    # Optional: Log the number of examples where the assistant token was not found
    print(f"Number of examples where assistant token was not found: {missing_assistant_token}")

    return {
        'input_ids': [x['input_ids'] for x in inputs],
        'attention_mask': [x['attention_mask'] for x in inputs],
        'labels': labels
    }

# Apply the preprocessing function to the datasets
print("Tokenizing the training dataset...")
tokenized_train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

print("Tokenizing the validation dataset...")
tokenized_validation_dataset = validation_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=validation_dataset.column_names,
)

# Filter out examples where the assistant token was not found
def filter_empty_labels(example):
    return any(label != -100 for label in example['labels'])

tokenized_train_dataset = tokenized_train_dataset.filter(filter_empty_labels)
tokenized_validation_dataset = tokenized_validation_dataset.filter(filter_empty_labels)

# Define a custom collate function to handle dynamic padding
def collate_fn(batch):
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

# Create DataLoaders
train_loader = DataLoader(
    tokenized_train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

validation_loader = DataLoader(
    tokenized_validation_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

# ------------------------------
# 3. Optimizer and Scheduler Setup
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

# Initialize mixed precision scaler using the updated API
scaler = torch.amp.GradScaler(enabled=fp16)

# ------------------------------
# 4. Evaluation Function
# ------------------------------

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            with torch.amp.autocast(device_type='cuda', enabled=fp16):
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss
                total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# ------------------------------
# 5. Training Loop
# ------------------------------

global_step = 0
best_eval_loss = float('inf')
steps_since_last_save = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for step, batch in enumerate(progress_bar):
        with torch.amp.autocast(device_type='cuda', enabled=fp16):
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        epoch_loss += loss.item() * gradient_accumulation_steps

        if (step + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimizer step with scaled gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            steps_since_last_save += 1

            # Logging
            if global_step % logging_steps == 0:
                print(f"Step {global_step}: Loss = {loss.item() * gradient_accumulation_steps:.4f}")

            # Evaluation
            if global_step % eval_steps == 0:
                eval_loss = evaluate(model, validation_loader)
                print(f"Evaluation at step {global_step}: Loss = {eval_loss:.4f}")
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    # Save the best model
                    model.save_pretrained(os.path.join(output_dir, "best_model"))
                    tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))
                    print(f"Best model saved at step {global_step}")
                steps_since_last_save = 0

            # Saving the model periodically
            if steps_since_last_save >= save_steps:
                checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                print(f"Checkpoint saved at step {global_step}")
                steps_since_last_save = 0

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")

    # End of epoch evaluation
    eval_loss = evaluate(model, validation_loader)
    print(f"End of Epoch {epoch + 1}: Validation Loss = {eval_loss:.4f}")
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        # Save the best model
        model.save_pretrained(os.path.join(output_dir, "best_model_epoch"))
        tokenizer.save_pretrained(os.path.join(output_dir, "best_model_epoch"))
        print(f"Best model updated at end of epoch {epoch + 1}")

# Save the final model
model.save_pretrained(os.path.join(output_dir, "final_model"))
tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
print(f"Training completed. Final model saved at {os.path.join(output_dir, 'final_model')}")
