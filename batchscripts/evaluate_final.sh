#!/bin/sh
#BSUB -J Evaluate
#BSUB -o logs/Eval%J.out
#BSUB -e logs/Eval%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5G]"
#BSUB -R "select[gpu32gb]"
#BSUB -W 6:00
#BSUB -N

#BSUB 
# end of BSUB options

module load cuda/11.8

source venv/bin/activate

python src/evaluation.py --model_dir models_final/Instruction/best_model_pretrain_christian --result_dir models_final/Instruction/best_model_pretrain_christian

#["nordjylland-news", "scandiqa-da", "scala-da"]