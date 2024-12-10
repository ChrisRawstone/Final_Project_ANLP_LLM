from copy import copy
import logging
from typing import Dict, List
from transformers import PreTrainedTokenizer
from datasets import concatenate_datasets


import matplotlib.pyplot as plt
from typing import Dict, List
from transformers import PreTrainedTokenizer, AutoTokenizer
from datasets import load_dataset
: from data.make_dataset import make_instruction_data
import numpy as np


# train_dataset, validation_dataset = make_instruction_data(data_openhermes=False, data_skolegpt=False, data_aya=True, shuffle=True)
dataset3 = load_dataset("CohereForAI/aya_dataset", "default")
dataset3 = dataset3['train'].filter(lambda example: example['language'] == 'Danish')
# dataset = concatenate_datasets([train_dataset, validation_dataset])
# print(dataset.shape)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Step 2: Define a Function to Tokenize Without Truncation and Collect Lengths
def collect_token_lengths(example):
    """
    Tokenizes the prompt without truncation and returns the token length.
    """
    instruction = example.get("instructions", "")
    input_text = example.get("inputs", "")
    output_text = example.get("outputs", "")
    
    user_prompt = f"<|user|>{instruction}\n{input_text}<|end_of_turn|>"
    assistant_prompt = f"<|assistant|>{output_text}<|end_of_turn|>"
    prompt = f"{user_prompt}{assistant_prompt}"
    
    tokens = tokenizer(prompt, truncation=False, padding=False)
    return {"token_length": len(tokens["input_ids"])}

# Step 3: Apply the Function to the Dataset to Collect Token Lengths
# concat train and validation dataset
token_length_dataset = dataset.map(
    collect_token_lengths,
    batched=False,
    remove_columns=dataset.column_names
)

# Extract the token lengths
all_lengths = token_length_dataset['token_length']

# Step 4: Plot the Histogram of Token Lengths
def plot_token_length_histogram(lengths, bins=50, max_length=None):
    print(sum(lengths))
    if max_length: # length is a list
        lengths = [length for length in lengths if length <= max_length]
    plt.figure(figsize=(12, 6))
    plt.hist(lengths, bins=bins, color='skyblue', edgecolor='black')
    plt.title('Histogram of Token Lengths', fontsize=20)
    plt.xlabel('Number of Tokens', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.xticks(fontsize=15)  
    plt.yticks(fontsize=15) 
    plt.grid(axis='y', alpha=0.75)
    plt.savefig("token_length_histogram.png")

if __name__ == "__main__":
    plot_token_length_histogram(all_lengths, bins=100, max_length=2000)

