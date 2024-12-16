#!/bin/sh
#BSUB -J scala-da
#BSUB -o logs/scala-da%J.out
#BSUB -e logs/scala-da%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2G]"
#BSUB -W 1:30
#BSUB -N

#BSUB 
# end of BSUB options

echo "scala-da Evaluation"

module load cuda/11.8

source venv/bin/activate

scandeval --model Qwen/Qwen2.5-0.5B --language da --dataset scala-da --num-iterations 10 --force