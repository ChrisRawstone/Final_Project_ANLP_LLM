from transformers import AutoModel

qwen2_name = "Qwen/Qwen2-0.5B"
qwen2_5_name = "Qwen/Qwen2.5-0.5B"

# Load models
qwen2 = AutoModel.from_pretrained(qwen2_name)
qwen2_5 = AutoModel.from_pretrained(qwen2_5_name)




# Inspect model layers
print(qwen2)
print(qwen2_5)


from transformers import AutoTokenizer

qwen2_tokenizer = AutoTokenizer.from_pretrained(qwen2_name)
qwen2_5_tokenizer = AutoTokenizer.from_pretrained(qwen2_5_name)

# Compare vocab size
print("Qwen2 Vocab Size:", len(qwen2_tokenizer))
print("Qwen2.5 Vocab Size:", len(qwen2_5_tokenizer))


from torchsummary import summary

summary(qwen2, input_size=(1, 128))  # Example input size
summary(qwen2_5, input_size=(1, 128))
