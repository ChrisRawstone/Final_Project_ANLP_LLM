import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('data/processed/text_data_statistics.csv')

# Create a figure with larger size
plt.figure(figsize=(15, 10))

# Create a bar plot showing total characters for each dataset
plt.subplot(2, 1, 1)
sns.barplot(data=df, x='Name', y='total_characters')
plt.xticks(rotation=45)
plt.title('Total Characters by Dataset')
plt.ylabel('Total Characters')

# Create a bar plot showing average sentence length
plt.subplot(2, 1, 2)
sns.barplot(data=df, x='Name', y='avg_sentence_length')
plt.xticks(rotation=45)
plt.title('Average Sentence Length by Dataset')
plt.ylabel('Average Characters per Sentence')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig('data/processed/dataset_statistics.png')

# Print tabular view
print("\nDataset Statistics:")
print(df[['Name', 'total_lines', 'avg_sentence_length', 'max_sentence_length']].to_string(index=False))
