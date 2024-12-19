from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


ds = load_dataset("alexandrainst/scandi-qa", "da")

print(ds)


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("final_model")

# Load the model
model = AutoModelForCausalLM.from_pretrained("final_model")

# Now you can use them, for example:
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model(**inputs)
