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
