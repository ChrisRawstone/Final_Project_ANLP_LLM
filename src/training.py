import os
import time
from copy import copy
from typing import Any, Dict, List, Optional, Tuple
import logging
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel
from tqdm import tqdm
import wandb

from utils import calculate_perplexity, save_model_checkpoint

# Configure the logger
logger = logging.getLogger(__name__)


def generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    prompts: List[str],
    max_new_tokens: int = 256,
) -> List[str]:
    """
    Generates responses for a list of prompts using sampling methods.

    Args:
        model (PreTrainedModel): The language model.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        device (torch.device): The device to run on.
        prompts (List[str]): List of prompt strings.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 256.

    Returns:
        List[str]: Generated responses.
    """
    model.eval()
    responses = []
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
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
            responses.append(response.strip())
    model.train()
    return responses

def validate_model(
    model: PreTrainedModel,
    val_loader: DataLoader,
    device: torch.device,
    fp16: bool,
) -> Tuple[float, float]:
    """
    Evaluates the model on the validation dataset.

    Args:
        model (PreTrainedModel): The model to evaluate.
        val_loader (DataLoader): The validation DataLoader.
        device (torch.device): The device to run on.
        fp16 (bool): Whether to use mixed precision.

    Returns:
        Tuple[float, float]: The validation loss and perplexity.
    """
    logger.info("Evaluating the model on the validation dataset...")
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=fp16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                val_loss += outputs.loss.item()
    val_loss /= len(val_loader)
    val_perplexity = calculate_perplexity(val_loss)
    logger.info(f"Validation Loss: {val_loss:.4f}, Validation Perplexity: {val_perplexity:.4f}")
    model.train()
    return val_loss, val_perplexity

def run_training_steps(
    model: PreTrainedModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    evaluation_prompts: List[str],
    tokenizer: PreTrainedTokenizer,
    num_epochs: int = 3,
    gradient_accumulation_steps: int = 4,
    fp16: bool = True,
    max_grad_norm: float = 1.0,
    output_dir: str = "models/checkpoint",
    save_steps: int = 1300,
) -> None:
    """
    Runs the training loop with evaluation and wandb logging.
    Saves the model every `save_steps` steps.

    Args:
        model (PreTrainedModel): The model to train.
        train_loader (DataLoader): The training DataLoader.
        val_loader (DataLoader): The validation DataLoader.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler.LambdaLR): The learning rate scheduler.
        scaler (Optional[torch.cuda.amp.GradScaler]): GradScaler for mixed precision.
        device (torch.device): The device to run on.
        evaluation_prompts (List[str]): Prompts for evaluation.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        num_epochs (int, optional): Number of epochs. Defaults to 3.
        gradient_accumulation_steps (int, optional): Gradient accumulation steps. Defaults to 4.
        fp16 (bool, optional): Whether to use mixed precision. Defaults to True.
        max_grad_norm (float, optional): Max gradient norm for clipping. Defaults to 1.0.
        output_dir (str, optional): Directory to save models. Defaults to "models/checkpoint".
        save_steps (int, optional): Steps interval to save models. Defaults to 25.
    """
    
    # Initialize W&B run and config and create table to store evaluation results
    run = wandb.init(project="your_project_name", job_type="train")
    evaluation_table = wandb.Table(columns=["Epoch", "Global Step", "Validation Loss", "Validation Perplexity", "Prompt", "Response"])

    model.train()
    logger.info("Starting training...")

    global_step = 0
    total_batches = 0

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        optimizer.zero_grad()

        start_time = time.time()

        for step, _batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            batch = {k: v.to(device) for k, v in _batch.items()}

            with torch.cuda.amp.autocast(enabled=fp16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / gradient_accumulation_steps

            if torch.isnan(loss):
                logger.error(f"Epoch {epoch + 1}, Step {step + 1}: Loss is nan. Exiting training.")
                return

            if fp16 and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * gradient_accumulation_steps
            total_batches += 1

            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16 and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                current_lr = scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / total_batches
                perplexity = calculate_perplexity(avg_loss)

                # Log training metrics to W&B
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "learning_rate": current_lr,
                        "training_loss": avg_loss,
                        "training_perplexity": perplexity,
                    },
                    step=global_step
                )

                # Perform evaluation and logging at specified intervals
                if global_step % save_steps == 0:
                    save_model_checkpoint(global_step, model, tokenizer, output_dir)
                    val_loss, val_perplexity = validate_model(
                        model, val_loader, device, fp16
                    )

                    # Log validation metrics to W&B
                    wandb.log(
                        {
                            "validation_loss": val_loss,
                            "validation_perplexity": val_perplexity,
                        },
                        step=global_step
                    )

                    # Evaluate prompts and collect responses
                    responses = generate_responses(model, tokenizer, device, evaluation_prompts)
                    for prompt, response in zip(evaluation_prompts, responses):
                        logger.info(f"\nPrompt: {prompt}\nResponse: {response}")
                        evaluation_table.add_data(
                            epoch + 1,
                            global_step,
                            val_loss,
                            val_perplexity,
                            prompt,
                            response
                        )

                # Log intermediate progress
                if (step + 1) % (10 * gradient_accumulation_steps) == 0:
                    avg_loss = epoch_loss / total_batches
                    elapsed_time = time.time() - start_time
                    logger.info(
                        f"Epoch {epoch + 1}, Step {step + 1}: Avg Loss = {avg_loss:.4f}, "
                        f"Elapsed Time = {elapsed_time:.2f}s"
                    )
                    start_time = time.time()

        # Handle any remaining gradients
        if (step + 1) % gradient_accumulation_steps != 0:
            if fp16 and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # Save the final model
    final_model_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    logger.info(f"Saving final model to {final_model_path}...")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info("Training completed successfully.")

    # Log the evaluation table to W&B after training
    run.log({"evaluation_responses": evaluation_table})

    # Finish the W&B run
    run.finish()