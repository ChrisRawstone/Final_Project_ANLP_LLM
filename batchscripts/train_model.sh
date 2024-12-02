#!/bin/sh
#BSUB -J Train
#BSUB -o logs/Train%J.out
#BSUB -e logs/Train%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5G]"
#BSUB -W 23:55
#BSUB -N

#BSUB 
# end of BSUB options

module load cuda/11.8

source venv_poetry/bin/activate

python src/train_model.py --max_length 512 --learning_rate 5e-7

# parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
# parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")  
# parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
# parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Steps for gradient accumulation")
    