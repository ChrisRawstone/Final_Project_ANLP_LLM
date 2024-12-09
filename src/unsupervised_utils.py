from copy import copy
import logging
from typing import Dict, List
from transformers import PreTrainedTokenizer

# Configure the logger
logger = logging.getLogger(__name__)

def unsupervised_preprocess_function(
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
    #assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    #missing_assistant_token = 0

    for input in examples["text"]:
        # add the end of turn token
        # TODO Is this the correct way to do for unsupervised data?
        # we need some kind of end of turn token, but if it should be this one I am not sure
        input = f"{input}<|end_of_turn|>"

        tokenized = tokenizer(
            input,
            truncation=True,
            max_length=max_length,
            padding=False, #TODO - This is set to false, but should it be true?
            return_attention_mask=True,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        # labels is just the input_ids
        labels_ids = copy(input_ids)
        # make the last label -100 since it is the end of the turn token
        labels_ids[-1] = -100

        inputs.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )
        labels.append(labels_ids)   

    return {
        "input_ids": [x["input_ids"] for x in inputs],
        "attention_mask": [x["attention_mask"] for x in inputs],
        "labels": labels,
    }
