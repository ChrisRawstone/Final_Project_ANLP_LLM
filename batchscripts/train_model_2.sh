#!/bin/sh
#BSUB -J train_instruction_2
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

python src/instruction_main.py \
--model_name models_final/Unsupervised/model_constant/final_model \
--batch_size 4 \
--num_epochs 1 \
--learning_rate 5e-06 \
--lr_scheduler constant \
--weight_decay 0.01 \
--max_length 512 \
--gradient_accumulation_steps 4 \
--fp16 True \
--max_grad_norm 1.0 \
--num_workers 4 \
--seed 42 \