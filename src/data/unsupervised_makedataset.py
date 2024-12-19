from tqdm import tqdm
from datasets import load_dataset,concatenate_datasets, Dataset,load_from_disk
from transformers import AutoTokenizer


#TODO - Right now a lot is hardcoded, make it more flexible. 
# For instance the max_token_length is hardcoded to 512
def make_dataset_unsupervised(path_to_raw_data: str, model_name: str, max_token_length: int = 512) -> Dataset:
    # Load the dataset
    dataset = load_dataset('text', data_files=path_to_raw_data,cache_dir="huggingface_cache")
    dataset = dataset['train']

    # make the save path    
    save_path = path_to_raw_data.replace("raw", "processed")
    # add the token length to the save path
    save_path = save_path.replace(".txt", f"_contextlength_{max_token_length}")    

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    combined_data = {"text": []}  # Adjust keys based on your dataset structure

    current_text = []

    # Use tqdm to add a progress bar
    for row in tqdm(dataset, desc="Processing rows"):
        # Tokenize the current row's text
        token_length = len(tokenizer(row["text"], truncation=False)["input_ids"])

        # If adding this row is eqaul to or exceeds the max token length, add the current_text to the combined_data
        # we add 2 to the max token length to account for the <|assistant|> token and the <|end_of_turn|> token
        if current_text and (len(tokenizer(" ".join(current_text), truncation=False)["input_ids"]) + token_length + 2) >= max_token_length:
            combined_data["text"].append(" ".join(current_text))        
            current_text = []            

        # Add the current row to the group
        current_text.append(row["text"])

    # Append any remaining data
    if current_text:
        combined_data["text"].append(" ".join(current_text))        

    # save the data to disk
    dataset = Dataset.from_dict(combined_data)      
    dataset.save_to_disk(save_path)
    # Create a new dataset
    #return dataset

def make_combined_data_set(data_set_names=["bookshop","cc100","culturax","dawiki","gigaword","medical","opensubtitles","reddit.da","twitter"],
                            max_token_length=512,
                            shuffle=True):
    # load all the datasets and concatenate them
    datasets = []
    for data_set_name in data_set_names:
        dataset = load_from_disk(f"data/processed/unsupervised/{data_set_name}/{data_set_name}_subset_contextlength_{max_token_length}")
        #dataset = dataset['train']
        datasets.append(dataset)
    
    combined_train = concatenate_datasets(datasets)
    # Shuffle data
    if shuffle:
        combined_train = combined_train.shuffle(seed=42)
    
     # Split into train and validation
    train_size = int(0.95 * len(combined_train))
    train_dataset = combined_train.select(range(train_size))
    val_dataset = combined_train.select(range(train_size, len(combined_train)))

    # save both the train and validation dataset to disk
    train_dataset.save_to_disk(f"data/processed/unsupervised/combined/combined_contextlength_{max_token_length}_train")
    val_dataset.save_to_disk(f"data/processed/unsupervised/combined/combined_contextlength_{max_token_length}_val")

if __name__ == "__main__":
    #paths = []
    paths = ["data/raw/unsupervised/bookshop/bookshop_subset.txt",
             "data/raw/unsupervised/cc100/cc100_subset.txt",
             "data/raw/unsupervised/culturax/culturaX_subset.txt",
             "data/raw/unsupervised/dawiki/dawiki_subset.txt",
             "data/raw/unsupervised/gigaword/gigaword_subset.txt",
             "data/raw/unsupervised/medical/medical_subset.txt",
             "data/raw/unsupervised/opensubtitles/opensubtitles_subset.txt",
             "data/raw/unsupervised/reddit.da/reddit.da_subset.txt",
             "data/raw/unsupervised/twitter/twitter_subset.txt",
    ]
    model_name = "Qwen/Qwen2.5-0.5B"
    for path in tqdm(paths, desc="Processing each dataset"):        
        make_dataset_unsupervised(path, model_name, max_token_length=512)

    # make the combined dataset
    make_combined_data_set()