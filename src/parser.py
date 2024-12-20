import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Training Configuration")
    # Add arguments with default values
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Path to the model")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="Use learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Steps for gradient accumulation")
    parser.add_argument("--fp16", type=bool, default=True, help="Enable mixed precision training")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--data_openhermes", type=bool, default=True, help="Use OpenHermes dataset")
    parser.add_argument("--data_skolegpt", type=bool, default=True, help="Use data_skolegpt dataset")
    parser.add_argument("--data_aya", type=bool, default=True, help="Use data_aya dataset")
    parser.add_argument("--shuffle", type=bool, default=True, help="Use data shuffle")
    parser.add_argument("--LoRA", type=bool, default=False, help="Use LoRA")
    return parser.parse_args()

def get_args_eval():
    parser = argparse.ArgumentParser(description="Eval Configuration")
    # Add arguments with default values
    parser.add_argument("--config_num", type=int, default=1, help="conf to use")
    return parser.parse_args()

#["nordjylland-news", "scandiqa-da", "scala-da"]