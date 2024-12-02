#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_model_lora.py

A script to load a trained LoRA-adapted and quantized causal language model,
and generate responses for given prompts.

Usage:
    python test_model_lora.py

Ensure that the trained model is available at the specified path.
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
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
    model_name = "Qwen/Qwen2.5-0.5B"
    model_path = "models/lora_first_iteration/final_model"  # Path to the saved model
    max_length = 512  # Adjust as needed
    fp16 = True  # Enable mixed precision

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add special tokens
    special_tokens_dict = {
        "additional_special_tokens": ["<|user|>", "<|assistant|>", "<|end_of_turn|>"]
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added_toks} special tokens to the tokenizer.")

    # Define the quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load the base model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Resize model embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Resized model embeddings to {len(tokenizer)} tokens.")

    # Load the LoRA adapted model
    model = PeftModel.from_pretrained(
        model,
        model_path,
    )

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
