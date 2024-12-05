

from datasets import load_dataset

ds = load_dataset("kobprof/skolegpt-instruct")

# save data 
ds.save_to_disk("data/instruct_data")