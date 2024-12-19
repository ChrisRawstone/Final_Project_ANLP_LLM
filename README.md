# Evaluating Instruction Tuning as an Standalone Approach for Adapting LLMs to Danish
Welcome to our github. Here are steps to reproduce our results:

## Environment:
Start by installing the packages in the requirements.txt file.
```
pip install -r requirements.txt
```

## Data requirements
### Unsupervised data:
We used a data set of 315 millions tokens which is a subset of a 13 billion token dataset provided by our Professor, Rob van der Goot.

Put the data in data/raw/unsupervised folder and run the following preprocessing commands to cut the data into smaller chunks:
```
python src/data/unsupervised_data_cut.py
```

Then run the following command to create the dataset in arrow format:
Remember to check potentially adjust paths in the script.

```
python src/data/unsupervised_makedataset.py
```

### Instruction data:
Run the following preprocessing commands:
```
python src/data/make_dataset.py
```

## Reproduce pre-trained + instruction tuning model
To reproduce the pre-trained + instruction tuning model, you need to:
* train the model on unsupervised data.
* train the model on instruction tuning data
* Evaluate on scandeval

#### Pre-training on unsupervised data:
```
src/unsupervised_main.py \
model_name: Qwen/Qwen2.5-0.5B
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
To reproduce the evaluation results:
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






