# Evaluating Instruction Tuning as an Standalone Approach for Adapting LLMs to Danish
Welcome to our github. Here are steps to reproduce our results:

## Environment:
Start by installing the packages in the requirements.txt file.
```
pip install -r requirements.txt
```

## GPU requirements
We used GPU's from DTU's HPC. The GPU's used had either 32gb VRAM (V100) or 40gb VRAM (A100).
Less video memory on GPU has not been tested. 

## Data requirements
### Unsupervised data:
For unsupervised data we used 13B Token dataset provided by our Professor, Rob van der Goot.

Put the data in data folder and (bla bla bla) and run the following preprocessing commands:
```
(Placeholder)
```


### Instruction data:
Put the data in data folder and (bla bla bla) and run the following preprocessing commands:
```
(Placeholder)
```



## Reproduce Model 2
To reproduce model 2 results: 
* train the model on unsupervised data.
* train the model on instruction tuning data
* Evaluate on scandeval

#### Continous Pretrain:
```
Write command here:
```


#### Instruction Train:
````
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
````


#### Evaluate on ScandEval: 
To reproduce the evaluation results for model 2
```
module load cuda/11.8

source venv/bin/activate

python src/evaluation.py --model_dir models_final/Instruction/best_model_pretrain_christian --result_dir models_final/Instruction/best_model_pretrain_christian
```


### Producing plots:

#### EM and F1 on ScandiQA across Training Steps
```
python src/plot_scandiqa_timeline_evaluation.py --plot_path result/instruction/final_plots/scandiqa_plot_model5.png --csv_path models_final/Instruction/best_model_instruct_christian/20241211094520/evaluation_scandiqa_results.csv
```






