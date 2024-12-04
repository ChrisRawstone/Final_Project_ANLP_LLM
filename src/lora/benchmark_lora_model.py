#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
benchmark_lora_model.py

A script to benchmark a LoRA-adapted and quantized causal language model using Scandeval.
"""

import logging
import torch
from pathlib import Path

# Ensure that the monkey_patch_scandeval.py is in the same directory or adjust the path accordingly
import monkey_patch_scandeval  # This applies the monkey patch

# Import Scandeval's Benchmarker after applying the monkey patch
try:
    from scandeval.benchmarker import Benchmarker
except ImportError:
    raise ImportError("Scandeval is not installed or not found in the Python path.")

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    # Configuration parameters
    base_model_dir = "Qwen/Qwen2.5-0.5B"  # Path to your base model and tokenizer
    lora_adapter_dir = "models/lora_first_iteration/final_model"  # Path to your LoRA adapters
    cache_dir = ".scandeval_cache"
    results_path = Path.cwd() / "scandeval_benchmark_results.jsonl"

    # Initialize the Benchmarker with desired parameters
    benchmarker = Benchmarker(
        progress_bar=True,
        save_results=True,
        task=None,  # Specify tasks if needed, e.g., "question_answering"
        dataset="scandiqa-da",  # Specify datasets if needed, e.g., "danish_qa_dataset"
        language=["da"],  # Danish language code
        model_language=["da"],  # Specify model languages if needed
        dataset_language=["da"],  # Specify dataset languages if needed
        framework=None,  # Auto-detect framework
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=16,
        evaluate_train=False,
        raise_errors=False,
        cache_dir=cache_dir,
        token=True,  # Assuming you're authenticated via Hugging Face CLI
        # load_in_4bit=True,  # Adjust based on your model's quantization
        use_flash_attention=True,  # Use Flash Attention if supported
        clear_model_cache=False,
        only_validation_split=True,
        few_shot=True,
        num_iterations=1,  # Set to 1 for initial benchmarking
        debug=False,
        run_with_cli=False,
    )

    # Load the tokenizer from the base model directory
    logger.info(f"Loading tokenizer from {base_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

    # Add special tokens
    special_tokens_dict = {
        "additional_special_tokens": ["<|user|>", "<|assistant|>", "<|end_of_turn|>"]
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added_toks} special tokens to the tokenizer.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Pad token was None. Set pad_token to eos_token.")

    # Load the base model
    logger.info(f"Loading base model from {base_model_dir}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        trust_remote_code=True,  # Adjust based on your needs
        device_map="auto",  # Automatically map layers to devices
    )

    # Resize model embeddings to match the tokenizer
    base_model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Resized model embeddings to {len(tokenizer)} tokens.")

    # Load the LoRA adapters
    logger.info(f"Loading LoRA adapters from {lora_adapter_dir}")
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_dir,  # Ensure that lora_adapter_dir contains the LoRA adapters
        device_map={"": "cuda"},  # Ensure correct device mapping
    )

    # Move model to the desired device (redundant if device_map is correctly set)
    lora_model.to("cuda")
    logger.info("LoRA model loaded and moved to device successfully.")

    # Proceed with benchmarking using lora_model and tokenizer
    logger.info("Running evaluation...")

    # Run the benchmark
    benchmark_results = benchmarker.benchmark(
        model=lora_model,    # Pass the PeftModel instance directly
        # tokenizer=tokenizer, # Pass the tokenizer with special tokens
    )

    # Save the results to the specified path
    with open(results_path, "w") as f:
        for result in benchmark_results:
            f.write(result.json(indent=4) + "\n")
            print(result.json(indent=4))

    logger.info(f"Benchmarking completed. Results saved to {results_path}.")

if __name__ == "__main__":
    main()
