from copy import copy
import logging
from typing import Dict, List
from transformers import PreTrainedTokenizer

# Configure the logger
logger = logging.getLogger(__name__)

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