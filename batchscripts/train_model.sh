#!/bin/sh
#BSUB -J train_instruction
#BSUB -o logs/Train%J.out
#BSUB -e logs/Train%J.err
#BSUB -q gpua100
#BSUB -n 16 
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2G]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -N
# end of BSUB options

module load cuda/11.8

source env-anlp/bin/activate

python src/instruction_main.py --batch_size 2 --num_epochs 1 --learning_rate 5e-8 --lr_scheduler cosine --weight_decay 0.01 --max_length 1024