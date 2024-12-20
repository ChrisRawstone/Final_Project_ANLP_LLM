#!/bin/sh
#BSUB -J train_instruction
#BSUB -o logs/Train%J.out
#BSUB -e logs/Train%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 16 
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2G]"
#BSUB -W 4:55
#BSUB -N
#BSUB 
# end of BSUB options

module load cuda/11.8

source venv/bin/activate

python src/instruction_main.py --batch_size 4 --num_epochs 1 --learning_rate 5e-6 --lr_scheduler constant --weight_decay 0.01 --max_length 512
