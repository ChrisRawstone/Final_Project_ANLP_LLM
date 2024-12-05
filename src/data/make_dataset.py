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
    # select the columns from the feature_map that is in the dataset
    dataset = dataset.map(lambda example: {k: example.get(k, None) for k in feature_map.keys()})
    dataset = dataset.map(
        lambda example: {new_feat: example[old_feat] for old_feat, new_feat in feature_map.items()},
        remove_columns=list(feature_map.keys()))
    return dataset

def make_instruction_data(data_openhermed: True, data_skolegpt: True, data_aya: True, shuffle=True): 
    # Load the datasets
    if data_openhermed:  
        dataset1 = load_dataset("Mabeck/danish-OpenHermes")
        dataset1 = dataset1['train']
    if data_skolegpt:
        dataset2 = load_dataset("kobprof/skolegpt-instruct")
        dataset2 = dataset2['train']
    if data_aya:
        dataset3 = load_dataset("CohereForAI/aya_dataset", "default")
        dataset3 = dataset3['train'].filter(lambda example: example['language'] == 'Danish')

    # Standardize features
    feature_map = {
        "system_prompt": "instructions",
        "question": "inputs",
        "response": "outputs", 
        "targets": "outputs"}
    
    standardized_datasets = []
    if data_openhermed:
        standardized_datasets.append(dataset1)
    if data_skolegpt:
        standardized_dataset2 = standardize_features(dataset2, feature_map)
        standardized_datasets.append(standardized_dataset2)
    if data_aya:
        standardized_dataset3 = standardize_features(dataset3, feature_map)
        standardized_datasets.append(standardized_dataset3)

    combined_train = concatenate_datasets(standardized_datasets)

    # Shuffle data
    if shuffle:
        combined_train = combined_train.shuffle(seed=42)
    
    # Columns to keep: instructions, inputs, outputs
    combined_train = combined_train.map(
    lambda example: {
        "instructions": example["instructions"],
        "inputs": example["inputs"],
        "outputs": example["outputs"]
    },
    remove_columns=combined_train.column_names  # Remove all columns not explicitly returned
    )

    # Split into train and validation
    train_size = int(0.95 * len(combined_train))
    train_dataset = combined_train.select(range(train_size))
    val_dataset = combined_train.select(range(train_size, len(combined_train)))
    
    return train_dataset, val_dataset

if __name__ == '__main__':
    train_dataset, val_dataset = make_instruction_data(data_openhermed=True, data_skolegpt=True, data_aya=True)