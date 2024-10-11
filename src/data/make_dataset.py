from datasets import load_dataset

if __name__ == '__main__':
    # Get the data and process it

    # Specify the dataset and the desired local directory
    dataset = load_dataset("KennethTM/eli5_question_answer_danish")

    # Step 2: Split the dataset into training and validation sets
    # You can adjust the 'test_size' parameter to control the size of the validation set
    split_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)

    # Step 3: Separate the splits
    train_dataset = split_dataset['train']
    validation_dataset = split_dataset['test']

    # Optional: Save the splits to disk if needed
    train_dataset.save_to_disk("data/raw/eli5_qa_danish/train")
    validation_dataset.save_to_disk("data/raw/eli5_qa_danish/validation")

    print("Training and validation datasets have been created and saved.")

    pass