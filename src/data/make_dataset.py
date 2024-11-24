from datasets import load_dataset, concatenate_datasets

def standardize_features(dataset, feature_map):
    """
    Standardize dataset features to match a unified format.
    Args:
        dataset: The dataset to be standardized.
        feature_map: A dictionary mapping original features to standardized features.
    Returns:
        The dataset with standardized features.
    """
    return dataset.map(
        lambda example: {new_feat: example[old_feat] for old_feat, new_feat in feature_map.items()},
        remove_columns=list(feature_map.keys())
    )

if __name__ == '__main__':
    # Load the datasets
    dataset1 = load_dataset("kobprof/skolegpt-instruct")
    dataset2 = load_dataset("Mabeck/danish-OpenHermes")

    # Standardize features
    dataset1_feature_map = {
        "system_prompt": "instructions",
        "question": "inputs",
        "response": "outputs"
    }
    standardized_dataset1 = standardize_features(dataset1['train'], dataset1_feature_map)

    # No changes needed for dataset2
    standardized_dataset2 = dataset2['train']

    # Concatenate the datasets
    combined_train = concatenate_datasets([standardized_dataset1, standardized_dataset2])

    # Print the number of rows for each dataset
    print(f"Rows in dataset1 (train): {len(dataset1['train'])}")
    print(f"Rows in dataset2 (train): {len(dataset2['train'])}")


    # Print the number of rows in the combined dataset
    print(f"Rows in combined dataset: {len(combined_train)}")

    # Split into train and validation
    train_size = int(0.95 * len(combined_train))
    train_dataset = combined_train.select(range(train_size))
    val_dataset = combined_train.select(range(train_size, len(combined_train)))

    # Save the train and validation datasets
    train_dataset.save_to_disk("data/processed/instruct_train_dataset")
    val_dataset.save_to_disk("data/processed/instruct_val_dataset")

    print("Datasets saved successfully!")
