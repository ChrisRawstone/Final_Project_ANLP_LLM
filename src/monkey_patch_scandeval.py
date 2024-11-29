# monkey_patch_scandeval.py

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Attempt to import Scandeval's classes
try:
    from scandeval.benchmarker import Benchmarker, BenchmarkResult
except ImportError:
    raise ImportError("Scandeval is not installed or not found in the Python path.")

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(message)s",
)
logger = logging.getLogger(__name__)

def _benchmark_single_patched(
    self,
    dataset_config,
    model_id: str,
    raise_errors: bool,
    model,
    tokenizer,
    benchmark_config,
):
    """Patched method to benchmark a single model on a single dataset, supporting LoRA models."""
    logger.info(f"Benchmarking {model_id} on {dataset_config.pretty_name}")

    try:
        # Determine device
        if benchmark_config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU specified but not available.")

        device = torch.device("cuda") if (benchmark_config.device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")
        logger.info(f"Using device: {device}")

        # Load the tokenizer
        logger.info(f"Loading tokenizer from {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Add special tokens
        special_tokens_dict = {
            "additional_special_tokens": ["<|user|>", "<|assistant|>", "<|end_of_turn|>"]
        }
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added_toks} special tokens to the tokenizer.")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Pad token was None. Set pad_token to eos_token.")

        # Load the base model with quantization if specified
        logger.info(f"Loading base model from {model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=benchmark_config.trust_remote_code,
            device_map="auto",  # Automatically map layers to devices
            load_in_4bit=benchmark_config.load_in_4bit,
            torch_dtype=torch.float16 if benchmark_config.load_in_4bit else None,
            llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offload for int8
        )

        # Resize model embeddings to match the tokenizer
        base_model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized model embeddings to {len(tokenizer)} tokens.")

        # Load the LoRA adapters
        logger.info(f"Loading LoRA adapters for {model_id}")
        lora_model = PeftModel.from_pretrained(
            base_model,
            model_id,  # Assuming the LoRA adapters are saved in the same directory
            device_map={"": device.type},  # Ensure correct device mapping
        )

        # Move model to the desired device (redundant if device_map is correctly set)
        lora_model.to(device)
        logger.info("Model moved to device successfully.")

        # Proceed with benchmarking using lora_model and tokenizer
        logger.info("Running evaluation...")

        # ... (Rest of your evaluation logic)

        # Mocked results for demonstration purposes
        results = {"accuracy": 0.90, "f1": 0.92, "perplexity": 15.67}
        metadata_dict = {
            "num_model_parameters": sum(p.numel() for p in lora_model.parameters()),
            "max_sequence_length": 512,
            "vocabulary_size": len(tokenizer),
            "generative": True,
            "few_shot": benchmark_config.few_shot,
            "validation_split": benchmark_config.only_validation_split,
        }

        logger.info(f"Evaluation results: {results}")
        logger.info(f"Metadata: {metadata_dict}")

        record = BenchmarkResult(
            dataset=dataset_config.name,
            task=dataset_config.task.name,
            dataset_languages=[language.code for language in dataset_config.languages],
            model=model_id,
            results=results,
            **metadata_dict,
        )
        logger.debug(f"Results:\n{results}")
        return record, lora_model, tokenizer

    except Exception as e:
        logger.error(f"Error benchmarking {model_id} on {dataset_config.pretty_name}: {e}")
        if raise_errors:
            raise e
        return {"error": str(e)}

# Apply the monkey patch
Benchmarker._benchmark_single = _benchmark_single_patched
logger.info("Successfully monkey patched Benchmarker._benchmark_single to support LoRA models.")
