#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_model.py

A script to load a Qwen2.5 causal language model and generate responses for given prompts.

Usage:
    python test_model.py

Ensure that the pretrained model is available at the specified path.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)

    # Check for GPU availability
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Configuration parameters
    model_name = "models/instruction/20241202155739/step_13000"
    max_length = 512  # Adjust as needed

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add special tokens
    special_tokens_dict = {
        "additional_special_tokens": ["<|user|>", "<|assistant|>", "<|end_of_turn|>"]
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added_toks} special tokens to the tokenizer.")

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Resize model embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Resized model embeddings to {len(tokenizer)} tokens.")

    model.to(device)

    # Ensure the tokenizer uses the special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare evaluation prompts
    evaluation_prompts = [
        "<|user|>Hvordan laver jeg en kop kaffe?<|end_of_turn|><|assistant|>",
        "<|user|>Hvad er meningen med livet?<|end_of_turn|><|assistant|>",
        "<|user|>Kan du forklare kvantemekanik?<|end_of_turn|><|assistant|>",
        # Add more prompts as needed
    ]

    # Generate responses
    model.eval()
    for prompt in evaluation_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                temperature=0.8,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
            # Decode only the generated tokens
            generated_tokens = output_ids[0][input_ids.shape[-1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"Prompt: {prompt}")
            print(f"Response: {response.strip()}\n")

if __name__ == "__main__":
    main()
