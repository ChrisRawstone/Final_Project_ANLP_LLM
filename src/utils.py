"""
utils.py

Utility functions for training the language model.
"""

import os
import time
import math
import random
import logging
import numpy as np
from copy import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel
from datasets import Dataset
from tqdm import tqdm
import wandb

# Configure the logger
logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    """
    Sets the seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_datasets(train_path: str, validation_path: str) -> Tuple[Dataset, Dataset]:
    """
    Loads training and validation datasets from disk.

    Args:
        train_path (str): Path to the training dataset.
        validation_path (str): Path to the validation dataset.

    Returns:
        Tuple[Dataset, Dataset]: Loaded training and validation datasets.
    """
    from datasets import load_from_disk

    train_dataset = load_from_disk(train_path)
    validation_dataset = load_from_disk(validation_path)
    return train_dataset, validation_dataset


def preprocess_function(
    examples: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
) -> Dict[str, List[List[int]]]:
    """
    Preprocesses the dataset by constructing prompts, tokenizing, and aligning labels.
    Only the assistant's response is used as labels; the rest are ignored (-100).

    Args:
        examples (Dict[str, List[str]]): A batch of examples from the dataset.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        max_length (int, optional): Maximum sequence length. Defaults to 256.

    Returns:
        Dict[str, List[List[int]]]: Tokenized inputs, attention masks, and labels.
    """
    inputs = []
    labels = []
    assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    missing_assistant_token = 0

    for instruction, input_text, output_text in zip(
        examples.get("instructions", [""] * len(examples["inputs"])),
        examples["inputs"],
        examples["outputs"],
    ):
        # Construct the prompt
        user_prompt = f"<|user|>{instruction}\n{input_text}<|end_of_turn|>"
        assistant_prompt = f"<|assistant|>{output_text}<|end_of_turn|>"
        prompt = f"{user_prompt}{assistant_prompt}"

        # Tokenize the prompt
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

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

        inputs.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )
        labels.append(labels_ids)

    # Log the number of examples where the assistant token was not found
    if missing_assistant_token > 0:
        logger.warning(
            f"Number of examples where assistant token was not found: {missing_assistant_token}"
        )

    return {
        "input_ids": [x["input_ids"] for x in inputs],
        "attention_mask": [x["attention_mask"] for x in inputs],
        "labels": labels,
    }


def filter_empty_labels(example: Dict[str, Any]) -> bool:
    """
    Filters out examples where all labels are -100.

    Args:
        example (Dict[str, Any]): A single example from the dataset.

    Returns:
        bool: True if the example has at least one label not equal to -100.
    """
    return any(label != -100 for label in example["labels"])


def create_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 2,
    num_workers: int = 4,
) -> DataLoader:
    """
    Creates a DataLoader with a custom collate function for dynamic padding.

    Args:
        dataset (Dataset): The tokenized dataset.
        tokenizer (PreTrainedTokenizer): The tokenizer used for padding.
        batch_size (int, optional): Batch size. Defaults to 2.
        num_workers (int, optional): Number of DataLoader workers. Defaults to 4.

    Returns:
        DataLoader: The DataLoader instance.
    """

    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function to handle dynamic padding for variable-length sequences.

        Args:
            batch (List[Dict[str, torch.Tensor]]): A batch of examples.

        Returns:
            Dict[str, torch.Tensor]: Padded input_ids, attention_mask, and labels.
        """
        input_ids = [torch.tensor(example["input_ids"]) for example in batch]
        attention_masks = [torch.tensor(example["attention_mask"]) for example in batch]
        labels = [torch.tensor(example["labels"]) for example in batch]

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
            "input_ids": input_ids_padded,
            "attention_mask": attention_masks_padded,
            "labels": labels_padded,
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


def evaluate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    prompts: List[str],
    max_new_tokens: int = 256,
) -> List[str]:
    """
    Generates responses for a list of prompts using sampling methods.

    Args:
        model (PreTrainedModel): The language model.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        device (torch.device): The device to run on.
        prompts (List[str]): List of prompt strings.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 256.

    Returns:
        List[str]: Generated responses.
    """
    model.eval()
    responses = []
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
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
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
            # Decode only the generated tokens
            generated_tokens = output_ids[0][input_ids.shape[-1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response.strip())
    model.train()
    return responses


def calculate_perplexity(loss: float) -> float:
    """
    Calculates perplexity from the loss value.

    Args:
        loss (float): The loss value.

    Returns:
        float: The perplexity.
    """
    try:
        return math.exp(loss)
    except OverflowError:
        logger.warning("Perplexity calculation resulted in overflow. Returning infinity.")
        return float("inf")


def save_model_checkpoint(
    step: int, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, output_dir: str
):
    """
    Save the model and tokenizer after resizing embeddings.

    Args:
        step (int): The current training step.
        model (PreTrainedModel): The model to save.
        tokenizer (PreTrainedTokenizer): The tokenizer to save.
        output_dir (str): The directory to save the model and tokenizer.
    """
    save_path = os.path.join(output_dir, f"step_{step}")
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Saving model at step {step} to {save_path}...")

    # Resize token embeddings before saving
    model.resize_token_embeddings(len(tokenizer))
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    logger.info(f"Model and tokenizer saved at step {step}.")


def evaluate_model(
    model: PreTrainedModel,
    val_loader: DataLoader,
    device: torch.device,
    fp16: bool,
) -> Tuple[float, float]:
    """
    Evaluates the model on the validation dataset.

    Args:
        model (PreTrainedModel): The model to evaluate.
        val_loader (DataLoader): The validation DataLoader.
        device (torch.device): The device to run on.
        fp16 (bool): Whether to use mixed precision.

    Returns:
        Tuple[float, float]: The validation loss and perplexity.
    """
    logger.info("Evaluating the model on the validation dataset...")
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=fp16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                val_loss += outputs.loss.item()
    val_loss /= len(val_loader)
    val_perplexity = calculate_perplexity(val_loss)
    logger.info(f"Validation Loss: {val_loss:.4f}, Validation Perplexity: {val_perplexity:.4f}")
    model.train()
    return val_loss, val_perplexity


def run_training_steps(
    model: PreTrainedModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    evaluation_prompts: List[str],
    tokenizer: PreTrainedTokenizer,
    num_epochs: int = 3,
    gradient_accumulation_steps: int = 4,
    fp16: bool = True,
    max_grad_norm: float = 1.0,
    num_steps_per_epoch: Optional[int] = None,
    output_dir: str = "models/checkpoint",
    save_steps: int = 1300,
) -> None:
    """
    Runs the training loop with evaluation and wandb logging.
    Saves the model every `save_steps` steps.

    Args:
        model (PreTrainedModel): The model to train.
        train_loader (DataLoader): The training DataLoader.
        val_loader (DataLoader): The validation DataLoader.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler.LambdaLR): The learning rate scheduler.
        scaler (Optional[torch.cuda.amp.GradScaler]): GradScaler for mixed precision.
        device (torch.device): The device to run on.
        evaluation_prompts (List[str]): Prompts for evaluation.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        num_epochs (int, optional): Number of epochs. Defaults to 3.
        gradient_accumulation_steps (int, optional): Gradient accumulation steps. Defaults to 4.
        fp16 (bool, optional): Whether to use mixed precision. Defaults to True.
        max_grad_norm (float, optional): Max gradient norm for clipping. Defaults to 1.0.
        num_steps_per_epoch (Optional[int], optional): Steps per epoch. Defaults to None.
        output_dir (str, optional): Directory to save models. Defaults to "models/checkpoint".
        save_steps (int, optional): Steps interval to save models. Defaults to 25.
    """
    # Initialize W&B run and config
    run = wandb.init(project="your_project_name", job_type="train")


    # Create a W&B table to store evaluation results
    evaluation_table = wandb.Table(columns=["Epoch", "Global Step", "Validation Loss", "Validation Perplexity", "Prompt", "Response"])

    model.train()
    logger.info("Starting training...")

    global_step = 0
    total_batches = 0

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        optimizer.zero_grad()

        start_time = time.time()

        for step, _batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            batch = {k: v.to(device) for k, v in _batch.items()}

            with torch.cuda.amp.autocast(enabled=fp16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / gradient_accumulation_steps

            if torch.isnan(loss):
                logger.error(f"Epoch {epoch + 1}, Step {step + 1}: Loss is nan. Exiting training.")
                return

            if fp16 and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * gradient_accumulation_steps
            total_batches += 1

            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16 and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                current_lr = scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / total_batches
                perplexity = calculate_perplexity(avg_loss)

                # Log training metrics to W&B
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "learning_rate": current_lr,
                        "training_loss": avg_loss,
                        "training_perplexity": perplexity,
                    },
                    step=global_step
                )

                # Perform evaluation and logging at specified intervals
                if global_step % save_steps == 0:
                    save_model_checkpoint(global_step, model, tokenizer, output_dir)
                    val_loss, val_perplexity = evaluate_model(
                        model, val_loader, device, fp16
                    )

                    # Log validation metrics to W&B
                    wandb.log(
                        {
                            "validation_loss": val_loss,
                            "validation_perplexity": val_perplexity,
                        },
                        step=global_step
                    )

                    # Evaluate prompts and collect responses
                    responses = evaluate(model, tokenizer, device, evaluation_prompts)
                    for prompt, response in zip(evaluation_prompts, responses):
                        logger.info(f"\nPrompt: {prompt}\nResponse: {response}")
                        evaluation_table.add_data(
                            epoch + 1,
                            global_step,
                            val_loss,
                            val_perplexity,
                            prompt,
                            response
                        )

                # Log intermediate progress
                if (step + 1) % (10 * gradient_accumulation_steps) == 0:
                    avg_loss = epoch_loss / total_batches
                    elapsed_time = time.time() - start_time
                    logger.info(
                        f"Epoch {epoch + 1}, Step {step + 1}: Avg Loss = {avg_loss:.4f}, "
                        f"Elapsed Time = {elapsed_time:.2f}s"
                    )
                    start_time = time.time()

        # Handle any remaining gradients
        if (step + 1) % gradient_accumulation_steps != 0:
            if fp16 and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # Save the final model
    final_model_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    logger.info(f"Saving final model to {final_model_path}...")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info("Training completed successfully.")

    # Log the evaluation table to W&B after training
    run.log({"evaluation_responses": evaluation_table})

    # Finish the W&B run
    run.finish()
