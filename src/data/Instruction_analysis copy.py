from datasets import load_dataset
import random

# Load datasets
print("=== Danish OpenHermes Dataset ===")
dataset1 = load_dataset("Mabeck/danish-OpenHermes")['train']
# This dataset likely has keys: "instructions", "inputs", "outputs"
print(f"Number of examples: {len(dataset1)}")
print("Sample Examples:")
for i in random.sample(range(len(dataset1)), 3):
    example = dataset1[i]
    print(f"\nExample {i}:")
    print("Instructions:", example.get('instructions', 'N/A'))
    print("Inputs:", example.get('inputs', 'N/A'))
    print("Outputs:", example.get('outputs', 'N/A'))


print("\n=== Skolegpt-instruct Dataset ===")
dataset2 = load_dataset("kobprof/skolegpt-instruct")['train']
# This dataset likely has keys: "system_prompt", "question", "response"
print(f"Number of examples: {len(dataset2)}")
print("Sample Examples:")
for i in random.sample(range(len(dataset2)), 3):
    example = dataset2[i]
    # Map system_prompt -> instructions, question -> inputs, response -> outputs for consistency
    print(f"\nExample {i}:")
    print("Instructions:", example.get('system_prompt', 'N/A'))
    print("Inputs:", example.get('question', 'N/A'))
    print("Outputs:", example.get('response', 'N/A'))


print("\n=== Aya Dataset (Danish only) ===")
dataset3 = load_dataset("CohereForAI/aya_dataset", "default")['train']
# Filter for Danish only
dataset3 = dataset3.filter(lambda ex: ex['language'] == 'Danish')
# This dataset likely has keys: "inputs", "targets"
print(f"Number of Danish examples: {len(dataset3)}")
print("Sample Examples:")
for i in random.sample(range(len(dataset3)), 3):
    example = dataset3[i]
    # Aya dataset doesn't explicitly have instructions; we only have inputs and targets.
    print(f"\nExample {i}:")
    print("Instructions: N/A (Not provided in this dataset)")
    print("Inputs:", example.get('inputs', 'N/A'))
    print("Outputs:", example.get('targets', 'N/A'))
