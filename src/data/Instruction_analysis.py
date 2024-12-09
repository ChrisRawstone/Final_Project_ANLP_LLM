    # Load and analyze the datasets
from datasets import load_dataset
import pandas as pd

# Load datasets
print("\nDanish OpenHermes Dataset:")

dataset1 = load_dataset("Mabeck/danish-OpenHermes")
dataset1 = dataset1['train']
print("Available keys:", dataset1[0].keys())
print(f"Number of examples: {len(dataset1)}")
print("Features:", dataset1.features)
#print("Sample row:", dataset1[0])

# Calculate total characters for dataset1
total_chars1 = sum(len(str(example['instructions'])) +len(str(example['inputs']))+ len(str(example['outputs'])) for example in dataset1)


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

# Calculate statistics
stats = pd.DataFrame({
    'Dataset': ['OpenHermes', 'Skolegpt', 'Aya'],
    'Examples': [len(dataset1), len(dataset2), len(dataset3)],
    'Total Characters': [total_chars1, total_chars2, total_chars3],
    'Avg Chars per Example': [total_chars1/len(dataset1), total_chars2/len(dataset2), total_chars3/len(dataset3)]
})

# Add totals row
totals = pd.DataFrame({
    'Dataset': ['Total'],
    'Examples': [stats['Examples'].sum()],
    'Total Characters': [stats['Total Characters'].sum()],
    'Avg Chars per Example': [stats['Total Characters'].sum() / stats['Examples'].sum()]
})

stats = pd.concat([stats, totals])

print("\nDataset Statistics:")
print(stats.to_string(index=False))