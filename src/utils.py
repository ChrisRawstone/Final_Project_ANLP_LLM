import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
)
from datasets import Dataset
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import wandb  # Import wandb
import math  # Import math for perplexity calculation

def set_seed(seed: int) -> None:
    """
    Sets the seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_datasets(
    train_path: str, validation_path: str
) -> (Dataset, Dataset):
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
    max_length: int = 256
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


def filter_empty_labels(example: Dict[str, Any]) -> bool:
    """
    Filters out examples where all labels are -100.

    Args:
        example (Dict[str, Any]): A single example from the dataset.

    Returns:
        bool: True if the example has at least one label not equal to -100.
    """
    return any(label != -100 for label in example['labels'])


def create_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 2,
    num_workers: int = 4
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

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )


def evaluate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    prompts: List[str],
    max_new_tokens: int = 256
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
                repetition_penalty=1.2,       # Added repetition penalty
                no_repeat_ngram_size=3,       # Prevent repeating n-grams of size 3
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
    return math.exp(loss)


def run_training_steps(
    model: PreTrainedModel,
    loader: DataLoader,
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
    output_dir: str = "models/checkpoint"  # Added output_dir parameter
) -> None:
    """
    Runs the training loop with evaluation and wandb logging.
    Saves the model after each epoch.

    Args:
        model (PreTrainedModel): The language model.
        loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler.LambdaLR): Learning rate scheduler.
        scaler (Optional[torch.cuda.amp.GradScaler]): GradScaler for mixed precision.
        device (torch.device): Device to run on.
        evaluation_prompts (List[str]): Prompts for evaluation after each epoch.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        num_epochs (int, optional): Number of training epochs. Defaults to 3.
        gradient_accumulation_steps (int, optional): Steps to accumulate gradients. Defaults to 4.
        fp16 (bool, optional): Whether to use mixed precision. Defaults to True.
        max_grad_norm (float, optional): Maximum gradient norm for clipping. Defaults to 1.0.
        num_steps_per_epoch (Optional[int], optional): Limit steps per epoch. Defaults to None.
        output_dir (str, optional): Directory to save the model checkpoints. Defaults to "models/checkpoint".
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
            with torch.amp.autocast(device_type="cuda",enabled=fp16):
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
            if fp16 and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate loss
            epoch_loss += loss.item() * gradient_accumulation_steps  # Multiply back to original loss

            # Optimizer step with gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16 and scaler is not None:
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

                # Log metrics to wandb
                current_lr = scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / ((step + 1) / gradient_accumulation_steps)
                perplexity = calculate_perplexity(avg_loss)
                wandb.log({
                    'epoch': epoch + 1,
                    'step': step + 1,
                    'learning_rate': current_lr,
                    'loss': avg_loss,
                    'perplexity': perplexity
                })

            # Log loss every 10 gradient accumulation steps
            if (step + 1) % (10 * gradient_accumulation_steps) == 0:
                avg_loss = epoch_loss / ((step + 1) / gradient_accumulation_steps)
                print(f"Epoch {epoch + 1}, Step {step + 1}: Avg Loss = {avg_loss:.4f}")

        # Handle remaining gradients if steps are not divisible by gradient_accumulation_steps
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

        # After each epoch, evaluate the model
        print("\nEvaluating the model with Danish prompts...")
        responses = evaluate(model, tokenizer, device, evaluation_prompts)

        # Log evaluation responses to wandb using a Table
        table = wandb.Table(columns=["Prompt", "Response"])
        for prompt, response in zip(evaluation_prompts, responses):
            print(f"\nPrompt: {prompt}\nResponse: {response}")
            table.add_data(prompt, response)


        # Log epoch metrics to wandb
        avg_epoch_loss = epoch_loss / (len(loader) / gradient_accumulation_steps)
        epoch_perplexity = calculate_perplexity(avg_epoch_loss)
        wandb.log({
            'epoch': epoch + 1,
            'avg_epoch_loss': avg_epoch_loss,
            'epoch_perplexity': epoch_perplexity,
            "evaluation_responses": table
        })

        # Save the model and tokenizer after each epoch
        epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
        os.makedirs(epoch_output_dir, exist_ok=True)
        print(f"\nSaving model to {epoch_output_dir}...")
        model.save_pretrained(epoch_output_dir)
        tokenizer.save_pretrained(epoch_output_dir)
        print(f"Model and tokenizer saved to {epoch_output_dir}.")

    print("\nTraining completed successfully.")
