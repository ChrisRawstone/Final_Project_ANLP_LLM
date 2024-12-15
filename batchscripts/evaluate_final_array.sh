#!/bin/sh
#BSUB -J Evaluate1[2-30:2]
#BSUB -o logs/Eval%J_%I.out
#BSUB -e logs/Eval%J_%I.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5G]"
#BSUB -W 1:15
#BSUB -N

#BSUB 
# end of BSUB options

module load cuda/11.8

source venv/bin/activate

python src/eval_scand_final.py --config_num $LSB_JOBINDEX

#["nordjylland-news", "scandiqa-da", "scala-da"]