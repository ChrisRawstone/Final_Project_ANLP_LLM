import os
import torch
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
)
from torch.utils.data import DataLoader
from typing import List, Optional
from tqdm import tqdm
import wandb
import math

def set_seed(seed: int) -> None:
    """
    Sets the seed for reproducibility.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_perplexity(loss: float) -> float:
    """
    Calculates perplexity from the loss value.
    """
    return math.exp(loss)

def evaluate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    prompts: List[str],
    max_new_tokens: int = 256
) -> List[str]:
    """
    Generates responses for a list of prompts using sampling methods.
    """
    model.eval()
    responses = []
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
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

def run_training_steps(
    model: PreTrainedModel,
    loader: DataLoader,
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
    num_steps_per_epoch: Optional[int] = None,
    output_dir: str = "models/checkpoint"
) -> None:
    """
    Runs the training loop with evaluation and wandb logging.
    Saves the model after each epoch.
    """
    model.train()
    print("\nStarting training...")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}")):
            if num_steps_per_epoch and step >= num_steps_per_epoch:
                break

            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with autocast
            with torch.cuda.amp.autocast(enabled=fp16):
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps  # Normalize loss

            # Check if loss is nan
            if torch.isnan(loss):
                print(f"Epoch {epoch + 1}, Step {step + 1}: Loss is nan. Exiting training.")
                return

            # Backward pass
            if fp16 and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate loss
            epoch_loss += loss.item() * gradient_accumulation_steps  # Multiply back to original loss

            # Optimizer step with gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16 and scaler is not None:
                    # Unscale gradients and clip
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Log metrics to wandb
                current_lr = scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / ((step + 1) / gradient_accumulation_steps)
                perplexity = calculate_perplexity(avg_loss)
                wandb.log({
                    'epoch': epoch + 1,
                    'step': step + 1,
                    'learning_rate': current_lr,
                    'loss': avg_loss,
                    'perplexity': perplexity
                })

            # Log loss every 10 gradient accumulation steps
            if (step + 1) % (10 * gradient_accumulation_steps) == 0:
                avg_loss = epoch_loss / ((step + 1) / gradient_accumulation_steps)
                print(f"Epoch {epoch + 1}, Step {step + 1}: Avg Loss = {avg_loss:.4f}")

        # Handle remaining gradients
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

        # After each epoch, evaluate the model
        print("\nEvaluating the model with Danish prompts...")
        responses = evaluate(model, tokenizer, device, evaluation_prompts)

        # Log evaluation responses to wandb using a Table
        table = wandb.Table(columns=["Prompt", "Response"])
        for prompt, response in zip(evaluation_prompts, responses):
            print(f"\nPrompt: {prompt}\nResponse: {response}")
            table.add_data(prompt, response)

        # Log epoch metrics to wandb
        avg_epoch_loss = epoch_loss / (len(loader) / gradient_accumulation_steps)
        epoch_perplexity = calculate_perplexity(avg_epoch_loss)
        wandb.log({
            'epoch': epoch + 1,
            'avg_epoch_loss': avg_epoch_loss,
            'epoch_perplexity': epoch_perplexity,
            "evaluation_responses": table
        })

        # Save the model and tokenizer after each epoch
        import datetime 
        model_name = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch + 1}_{model_name}")
        os.makedirs(epoch_output_dir, exist_ok=True)
        print(f"\nSaving model to {epoch_output_dir}...")
        model.save_pretrained(epoch_output_dir)
        tokenizer.save_pretrained(epoch_output_dir)
        print(f"Model and tokenizer saved to {epoch_output_dir}.")

    print("\nTraining completed successfully.")
