# Reproducing results:


To reproduce the evaluation results for model 2
```
module load cuda/11.8

source venv/bin/activate

python src/evaluation.py --model_dir models_final/Instruction/best_model_pretrain_christian --result_dir models_final/Instruction/best_model_pretrain_christian
```
