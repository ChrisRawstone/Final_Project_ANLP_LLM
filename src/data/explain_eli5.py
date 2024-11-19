from datasets import load_from_disk

# Define paths to datasets
train_path = "data/raw/eli5_qa_danish/train"
validation_path = "data/raw/eli5_qa_danish/validation"

# Load the datasets from disk
train_dataset = load_from_disk(train_path)
validation_dataset = load_from_disk(validation_path)

# Function to filter dataset based on all keywords in query or passage
def filter_by_all_keywords(dataset, keywords):
    def contains_all_keywords(example):
        query = example['query'].lower()
        # Uncomment the line below if you want to filter by passage as well
        # passage = example['passage'].lower()
        return all(keyword.lower() in query for keyword in keywords)
               # and all(keyword.lower() in passage for keyword in keywords)

    filtered_dataset = dataset.filter(contains_all_keywords)
    return filtered_dataset

# Function to print the dataset in a formatted way
def print_filtered_results(dataset, name):
    print(f"\n--- {name} ---\n")
    for i, example in enumerate(dataset):
        print(f"Result {i+1}:")
        print(f"Query: {example['query']}")
        print(f"Passage: {example['passage']}")
        print("-" * 50)

# List of keywords that must all be present
keywords = ["vin","øl","kaffe"]

# Filter the datasets for all the keywords
filtered_train = filter_by_all_keywords(train_dataset, keywords)
filtered_validation = filter_by_all_keywords(validation_dataset, keywords)

# Print the filtered results in a formatted way
print_filtered_results(filtered_train, "Filtered Train Dataset")
print_filtered_results(filtered_validation, "Filtered Validation Dataset")
