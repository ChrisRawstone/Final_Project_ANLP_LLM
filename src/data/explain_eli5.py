from datasets import load_dataset


# Load the dataset
dataset = load_dataset("KennethTM/eli5_question_answer_danish")

# Print columns and format for each split in the dataset (train, validation, test)
for split in dataset:
    print(f"Split: {split}")
    # Print column names
    print("Columns:", dataset[split].column_names)
    # Print an example row to show the format
    print("Example Row:", dataset[split][0])






    