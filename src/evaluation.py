"""
evaluation.py

Evaluating LLM model on ScandiQA-DA dataset
"""

from scandeval import Benchmarker
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

# Load the base model and tokenizer with mismatched size handling
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    ignore_mismatched_sizes=True,  # Ignore size mismatches
    torch_dtype="auto",           # Automatically pick appropriate dtype
    low_cpu_mem_usage=True,       # Reduce memory usage during loading
)

tokenizer = AutoTokenizer.from_pretrained("models/lora_first_iteration/step_1300")

# Resize token embeddings to match the tokenizer's vocabulary size
base_model.resize_token_embeddings(len(tokenizer))

# Load the LoRA adapter
lora_model = PeftModel.from_pretrained(
    base_model,
    "models/lora_first_iteration/step_1300",
    torch_dtype="float16",  # Explicitly set to float16
)

# Save the fully adapted model
lora_model.save_pretrained("models/lora_first_iteration/adapted_lora")
tokenizer.save_pretrained("models/lora_first_iteration/adapted_lora")

# Initialize the Benchmarker
benchmark = Benchmarker(
    progress_bar=True,
    save_results=True,
    device="cuda",
    verbose=True,
    num_iterations=3,
)

# Run the benchmark with explicit dtype settings
results = benchmark(
    model="models/lora_first_iteration/adapted_lora",
    dataset="scandiqa-da",
    language="da",
    framework="pytorch",
    trust_remote_code=True,
    device="cuda",
    verbose=True,
)

# Print results
for result in results:
    print(json.dumps(result, indent=2))
