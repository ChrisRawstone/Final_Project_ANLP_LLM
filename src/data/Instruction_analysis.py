    # Load and analyze the datasets
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_instruction import preprocess_function

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
special_tokens_dict = {'additional_special_tokens': ['<|user|>', '<|assistant|>', '<|end_of_turn|>']}
tokenizer.add_special_tokens(special_tokens_dict)

# Load datasets
print("\nDanish OpenHermes Dataset:")

dataset1 = load_dataset("Mabeck/danish-OpenHermes")
dataset1 = dataset1['train']
print("Available keys:", dataset1[0].keys())
print(f"Number of examples: {len(dataset1)}")
print("Features:", dataset1.features)
#print("Sample row:", dataset1[0])

# Calculate total characters for dataset1
total_chars1 = sum(len(str(example['instructions'])) + len(str(example['inputs'])) + len(str(example['outputs'])) for example in dataset1)


print("\nSkolegpt-instruct Dataset:")

dataset2 = load_dataset("kobprof/skolegpt-instruct")
print("Available keys:", dataset2.keys())
dataset2 = dataset2['train']
print(f"Number of examples: {len(dataset2)}")
print("Features:", dataset2.features)
#print("Sample row:", dataset2[0])

# Calculate total characters for dataset2

total_chars2 = sum(len(str(example['system_prompt'])) + len(str(example['question'])) + len(str(example['response'])) for example in dataset2)





print("\nAya Dataset (Danish only):")

dataset3 = load_dataset("CohereForAI/aya_dataset", "default")

dataset3 = dataset3['train'].filter(lambda example: example['language'] == 'Danish')
print("print dataset:", dataset3)

print(f"Number of examples: {len(dataset3)}")
print("Features:", dataset3.features)
#print("Sample row:", dataset3[0])

# Calculate total characters for dataset3
total_chars3 = sum(len(str(example['inputs'])) + len(str(example['targets'])) for example in dataset3)

# Function to count tokens for each component
def count_component_tokens(dataset, name):
    print(f"\nToken counts for {name}:")
    
    # Convert dataset to dictionary format for preprocess_function
    dataset_dict = {
        "instructions": [str(example.get('instructions', '') or example.get('system_prompt', '') or '') for example in dataset],
        "inputs": [str(example.get('inputs', '') or example.get('question', '') or '') for example in dataset],
        "outputs": [str(example.get('outputs', '') or example.get('response', '') or example.get('targets', '') or '') for example in dataset]
    }
    
    # Apply preprocessing
    preprocessed = preprocess_function(dataset_dict, tokenizer)
    
    # Count tokens from preprocessed data
    total_input_tokens = sum(sum(1 for x in ids if x != -100) for ids in preprocessed['input_ids'])
    total_output_tokens = sum(sum(1 for x in labels if x != -100) for labels in preprocessed['labels'])
    
    print(f"Input tokens (instructions + inputs): {total_input_tokens:,}")
    print(f"Output tokens: {total_output_tokens:,}")
    print(f"Total tokens: {total_input_tokens + total_output_tokens:,}")
    print(f"Average tokens per example: {(total_input_tokens + total_output_tokens) / len(dataset):,.1f}")
    
    return total_input_tokens, total_output_tokens

# Test different context lengths
for max_len in [256, 512, 1024, 2048]:
    print(f"\n=== Testing max_length = {max_len} ===")
    tokens1 = count_component_tokens(dataset1, "OpenHermes")
    tokens2 = count_component_tokens(dataset2, "Skolegpt")
    tokens3 = count_component_tokens(dataset3, "Aya")
    
    # Print total tokens across all datasets
    total_inputs = sum(t[0] for t in [tokens1, tokens2, tokens3])
    total_outputs = sum(t[1] for t in [tokens1, tokens2, tokens3])
    total_examples = len(dataset1) + len(dataset2) + len(dataset3)
    
    print(f"\nTotal tokens across all datasets (max_length={max_len}):")
    print(f"Total input tokens (instructions + inputs): {total_inputs:,}")
    print(f"Total output tokens: {total_outputs:,}")
    print(f"Total combined tokens: {total_inputs + total_outputs:,}")
    print(f"Average tokens per example: {(total_inputs + total_outputs) / total_examples:,.1f}")