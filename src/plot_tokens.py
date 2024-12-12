from copy import copy
import logging
from typing import Dict, List
from transformers import PreTrainedTokenizer
from datasets import concatenate_datasets


import matplotlib.pyplot as plt
from typing import Dict, List
from transformers import PreTrainedTokenizer, AutoTokenizer
from datasets import load_dataset
from data.make_dataset import make_instruction_data
import numpy as np


train_dataset, validation_dataset = make_instruction_data(data_openhermes=True, data_skolegpt=True, data_aya=True, shuffle=True)
# dataset3 = load_dataset("CohereForAI/aya_dataset", "default")
# dataset3 = dataset3['train'].filter(lambda example: example['language'] == 'Danish')
dataset = concatenate_datasets([train_dataset, validation_dataset])
# print(dataset.shape)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Step 2: Define a Function to Tokenize Without Truncation and Collect Lengths
def collect_token_lengths_input(example):
    """
    Tokenizes the prompt without truncation and returns the token length.
    """
    instruction = example.get("instructions", "")
    input_text = example.get("inputs", "")
    
    user_prompt = f"{instruction}\n{input_text}"
    #assistant_prompt = f"{output_text}
    # prompt = f"{user_prompt}"
    
    tokens = tokenizer(user_prompt, truncation=False, padding=False)
    return {"token_length": len(tokens["input_ids"])}

def collect_token_lengths_output(example):
    """
    Tokenizes the prompt without truncation and returns the token length.
    """
    output_text = example.get("outputs", "")    
    tokens = tokenizer(output_text, truncation=False, padding=False)
    return {"token_length": len(tokens["input_ids"])}


# Step 3: Apply the Function to the Dataset to Collect Token Lengths
# concat train and validation dataset
selected_columns = ['inputs', 'instructions']
dataset_subs = dataset.select_columns(selected_columns)
token_length_dataset = dataset_subs.map(
    collect_token_lengths_input,
    batched=False,
    remove_columns=dataset_subs.column_names
)

selected_columns = ['outputs']
dataset_subs = dataset.select_columns(selected_columns)
token_length_dataset = dataset_subs.map(
    collect_token_lengths_output,
    batched=False,
    remove_columns=dataset_subs.column_names
)

# Extract the token lengths
all_lengths = token_length_dataset['token_length']




