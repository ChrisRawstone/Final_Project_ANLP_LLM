"""
Main script to train a causal language model on Danish question-answering data.
"""
# ------------------------------
# 1. Imports and Configuration
# ------------------------------
import os
from datetime import datetime
import logging

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
)
import wandb
wandb.login(key="83fb1d160dc4cb3bbaceadab26cba368ebced6c6")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import (
    set_seed,
    create_dataloader, 
    filter_empty_labels)
from utils_instruction import preprocess_function
from training import run_training_steps
from evaluation import evaluate_scandeval
from data.make_dataset import make_instruction_data
from parser import get_args

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main(args) -> None:
    model_name = "Qwen/Qwen2.5-0.5B"
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    lr_scheduler = args.lr_scheduler
    weight_decay = args.weight_decay
    max_length = args.max_length
    gradient_accumulation_steps = args.gradient_accumulation_steps
    fp16 = args.fp16
    max_grad_norm = args.max_grad_norm
    num_workers = args.num_workers
    seed = args.seed
    data_openhermes = args.data_openhermes
    data_skolegpt = args.data_skolegpt
    data_aya = args.data_aya
    shuffle = args.shuffle
    
    # ------------------------------
    # 2. Set Up Experiment
    # ------------------------------
    
    # Set random seeds for reproducibility    
    set_seed(seed)
    
    # Create a timestamp for the output directory and save the configuration
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_dir = f"models/instruction/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    # save the configuration
    with open(f"{output_dir}/config.txt", "w") as f:
        f.write(f"model_name: {model_name}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"num_epochs: {num_epochs}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"lr_scheduler: {lr_scheduler}\n")
        f.write(f"weight_decay: {weight_decay}\n")
        f.write(f"max_length: {max_length}\n")
        f.write(f"gradient_accumulation_steps: {gradient_accumulation_steps}\n")
        f.write(f"fp16: {fp16}\n")
        f.write(f"max_grad_norm: {max_grad_norm}\n")
        f.write(f"num_workers: {num_workers}\n")
        f.write(f"seed: {seed}\n")

    # Check for GPU availability
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize wandb
    """wandb.init(
        project="danish-qa-model",
        name=f"instruction-{timestamp}",
        config={
            "model_name": model_name,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "lr_scheduler": lr_scheduler,
            "weight_decay": weight_decay,
            "max_length": max_length,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "fp16": fp16,
            "max_grad_norm": max_grad_norm,
            "num_workers": num_workers,
            "seed": seed,
            "output_dir": output_dir,
        },
        reinit=True)"""

    # ------------------------------
    # 3. Load Tokenizer and Model
    # ------------------------------

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add special tokens
    special_tokens_dict = {'additional_special_tokens': ['<|user|>', '<|assistant|>', '<|end_of_turn|>']}
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
    # 4. Load Data
    # ------------------------------

    # Load the datasets from disk
    print("Getting data...")
    train_dataset, validation_dataset = make_instruction_data(data_openhermes=data_openhermes, data_skolegpt=data_skolegpt, data_aya=data_aya, shuffle=shuffle)

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
    total_train_user_tokens = 0
    total_train_assistant_tokens = 0
    
    def accumulate_tokens(examples):
        result = preprocess_function(examples, tokenizer, max_length)
        nonlocal total_train_user_tokens, total_train_assistant_tokens
        total_train_user_tokens += result['batch_user_tokens'][0]  # Extract from list
        total_train_assistant_tokens += result['batch_assistant_tokens'][0]  # Extract from list
        return result

    tokenized_train_dataset = small_train_dataset.map(
        accumulate_tokens,
        batched=True,
        remove_columns=small_train_dataset.column_names,
        keep_in_memory=True)
    
    print("\nTraining Dataset Token Counts (excluding special tokens):")
    print(f"Total user tokens: {total_train_user_tokens}")
    print(f"Total assistant tokens: {total_train_assistant_tokens}")
    print(f"Total combined tokens: {total_train_user_tokens + total_train_assistant_tokens}")

    # Preprocess the validation dataset
    print("\nPreprocessing the validation dataset...")
    total_val_user_tokens = 0
    total_val_assistant_tokens = 0
    
    def accumulate_val_tokens(examples):
        result = preprocess_function(examples, tokenizer, max_length)
        nonlocal total_val_user_tokens, total_val_assistant_tokens
        total_val_user_tokens += result['batch_user_tokens'][0]  # Extract from list
        total_val_assistant_tokens += result['batch_assistant_tokens'][0]  # Extract from list
        return result

    tokenized_val_dataset = validation_dataset.map(
        accumulate_val_tokens,
        batched=True,
        remove_columns=validation_dataset.column_names,
        keep_in_memory=True)
    
    print("\nValidation Dataset Token Counts (excluding special tokens):")
    print(f"Total user tokens: {total_val_user_tokens}")
    print(f"Total assistant tokens: {total_val_assistant_tokens}")
    print(f"Total combined tokens: {total_val_user_tokens + total_val_assistant_tokens}")

    # Print example structure and token counts
    print("\nInspecting first example in tokenized dataset:")
    first_example = tokenized_train_dataset[0]
    
    # Initialize counters for total tokens and characters in training dataset
    total_instruction_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_instruction_chars = 0
    total_input_chars = 0
    total_output_chars = 0
    
    # Count tokens and characters for training dataset
    for example in small_train_dataset:
        # Use standardized field names
        instruction = str(example.get('instructions', '') or '')
        input_text = str(example.get('inputs', '') or '')
        output_text = str(example.get('outputs', '') or '')
        
        # Count tokens and characters
        instruction_tokens = len(tokenizer.encode(instruction))
        input_tokens = len(tokenizer.encode(input_text))
        output_tokens = len(tokenizer.encode(output_text))
        
        total_instruction_tokens += instruction_tokens
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        
        total_instruction_chars += len(instruction)
        total_input_chars += len(input_text)
        total_output_chars += len(output_text)
    
    # Print token counts for training dataset
    print(f"\nTraining Dataset Token Counts:")
    print(f"Total instruction tokens: {total_instruction_tokens}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total combined tokens: {total_instruction_tokens + total_input_tokens + total_output_tokens}")
    
    # Print character counts for training dataset
    print(f"\nTraining Dataset Character Counts:")
    print(f"Total combined characters: {total_instruction_chars + total_input_chars + total_output_chars}")
    

    
    # Initialize counters for total tokens and characters in validation dataset
    total_val_instruction_tokens = 0
    total_val_input_tokens = 0
    total_val_output_tokens = 0
    total_val_instruction_chars = 0
    total_val_input_chars = 0
    total_val_output_chars = 0
    
    # Count tokens and characters for validation dataset
    for example in validation_dataset:
        instruction = example.get('instructions', '') or ''
        input_text = example.get('inputs', '') or ''
        output_text = example.get('outputs', '') or ''
        
        # Count tokens
        instruction_tokens = len(tokenizer.encode(instruction))
        input_tokens = len(tokenizer.encode(input_text))
        output_tokens = len(tokenizer.encode(output_text))
        
        total_val_instruction_tokens += instruction_tokens
        total_val_input_tokens += input_tokens
        total_val_output_tokens += output_tokens
        
        # Count characters
        total_val_instruction_chars += len(instruction)
        total_val_input_chars += len(input_text)
        total_val_output_chars += len(output_text)
    
    # Print token counts for validation dataset
    print(f"\nValidation Dataset Token Counts:")
    print(f"Total instruction tokens: {total_val_instruction_tokens}")
    print(f"Total input tokens: {total_val_input_tokens}")
    print(f"Total output tokens: {total_val_output_tokens}")
    print(f"Total combined tokens: {total_val_instruction_tokens + total_val_input_tokens + total_val_output_tokens}")
    
    # Print character counts for validation dataset
    print(f"\nValidation Dataset Character Counts:")
    print(f"Total combined characters: {total_val_instruction_chars + total_val_input_chars + total_val_output_chars}")

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

    # Calculate total training steps (steps where parameters are updated, this is the number of backward passes)
    total_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize the learning rate scheduler
    if lr_scheduler: 
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,  # 10% of total steps for warm-up
            num_training_steps=total_steps
        )
    else: 
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10  # 10% of total steps for warm-up
        )

    # Initialize GradScaler if using mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=fp16) if fp16 else None

    # ------------------------------
    # 8. Prepare Evaluation Prompts
    # ------------------------------
    evaluation_prompts = [
        "<|user|>Hvordan laver jeg en kop kaffe?<|end_of_turn|>",
        "<|user|>Hvad er meningen med livet?<|end_of_turn|>",
        "<|user|>Kan du forklare kvantemekanik?<|end_of_turn|>",
    ]

    # ------------------------------
    # 9. Run the Training
    # ------------------------------
    """ run_training_steps(
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
        output_dir=output_dir
    )"""

    print("\nTraining script completed.")

    wandb.finish() # Finish the wandb run

    evaluate_scandeval(MODEL_DIR=output_dir, RESULT_DIR=f"result/instruction/{timestamp}")

if __name__ == "__main__":
    args = get_args()
    main(args)