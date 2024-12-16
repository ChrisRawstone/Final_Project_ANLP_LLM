import os
import pandas as pd
from scandeval import Benchmarker
from parser import get_args_eval
import logging

# Define a mapping from dataset names to their respective metrics
DATASET_METRICS = {
    "scandiqa-da": ["test_em", "test_f1", "test_em_se", "test_f1_se"],
    "scala-da": ["test_mcc", "test_macro_f1", "test_mcc_se", "test_macro_f1_se"],
    "nordjylland-news": ["test_bertscore", "test_rouge_l", "test_bertscore_se", "test_rouge_l_se"]
}

def get_final_model_path(model_dir):
    """
    Retrieves the path to the 'final_model' directory within the specified model directory.
    """
    final_model_dir = os.path.join(model_dir, "final_model")
    if os.path.isdir(final_model_dir):
        return final_model_dir
    else:
        raise FileNotFoundError(f"'final_model' not found in {model_dir}.")

def ensure_directory(dir_path):
    """Ensures that the specified directory exists. Creates it if it doesn't."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

def evaluate_final_model(
    MODEL_DIR,
    RESULT_DIR,
    DATASETS=["scandiqa-da", "scala-da", "nordjylland-news"],
    LANGUAGE="da",
    FRAMEWORK="pytorch",
    DEVICE="cuda",
    NUM_ITERATIONS=10,
    DEBUG=False,
):
    """
    Evaluates the 'final_model' on specified datasets and saves the results to individual CSV files named after each benchmark.
    Accommodates different evaluation metrics based on the dataset.
    """
    # Ensure result directory exists
    ensure_directory(RESULT_DIR)

    try:
        # Get the final model path
        final_model_path = get_final_model_path(MODEL_DIR)
        model_name = os.path.basename(final_model_path)

        print(f"\nEvaluating model: {model_name} on datasets: {DATASETS}\n")

        # # Initialize the Benchmarker
        # benchmark = Benchmarker(
        #     progress_bar=True,
        #     save_results=False,  # We handle saving ourselves
        #     device=DEVICE,
        #     verbose=True,
        #     num_iterations=NUM_ITERATIONS,
        #     force=True,
        #     debug=False # if this is true it seems the outputs are saved to file
        # )

        for dataset in DATASETS:

            # Initialize the Benchmarker, # didnt change anything moving it here
            benchmark = Benchmarker(
                progress_bar=True,
                save_results=False,  # We handle saving ourselves
                device=DEVICE,
                verbose=True,
                num_iterations=NUM_ITERATIONS,
                force=True,
                debug=DEBUG # if this is true it seems the outputs are saved to file
            )
            

            print(f"--- Benchmarking on dataset: {dataset} ---\n")
            try:
                # Run the benchmark and capture the results
                results = benchmark(
                    model=final_model_path,
                    dataset=dataset,
                    language=LANGUAGE,
                    framework=FRAMEWORK,
                    trust_remote_code=True,
                    verbose=True,
                )
                print(f"Type of results for dataset '{dataset}': {type(results)}")
                if isinstance(results, list):
                    print(f"Number of results returned: {len(results)}")
                else:
                    print(f"Results content: {results}")

                # Initialize variables
                extracted_metrics = {metric: None for metric in DATASET_METRICS.get(dataset, [])}

                # Process results based on dataset-specific metrics
                if dataset not in DATASET_METRICS:
                    print(f"No metric mapping defined for dataset '{dataset}'. Skipping metric extraction.\n")
                else:
                    metrics = DATASET_METRICS[dataset]
                    # Process results
                    if isinstance(results, list):
                        # Assuming 'results' is a list of BenchmarkResult objects
                        # Extract 'total' from the last result to get aggregated metrics
                        last_result = results[-1]
                        if hasattr(last_result, 'results') and isinstance(last_result.results, dict):
                            total = last_result.results.get('total', {})
                            for metric in metrics:
                                extracted_metrics[metric] = total.get(metric, None)
                        else:
                            print(f"Unexpected result format for model {model_name} on dataset {dataset}.")
                    elif hasattr(results, 'results'):
                        # Single BenchmarkResult object
                        total = results.results.get('total', {})
                        for metric in metrics:
                            extracted_metrics[metric] = total.get(metric, None)
                    else:
                        print(f"Unexpected result format for model {model_name} on dataset {dataset}.")

                # Debugging: Print extracted metrics
                metrics_str = ", ".join([f"{k}={v}" for k, v in extracted_metrics.items()])
                print(f"Extracted metrics for {model_name} on {dataset}: {metrics_str}\n")

                # Prepare the result entry
                result_entry = {
                    "model": model_name,
                    "dataset": dataset
                }
                result_entry.update(extracted_metrics)

                # Create a DataFrame for the current dataset
                df_dataset = pd.DataFrame([result_entry])

                # Ensure all possible metric columns are present in the DataFrame
                all_possible_metrics = DATASET_METRICS.get(dataset, [])
                for metric in all_possible_metrics:
                    if metric not in df_dataset.columns:
                        df_dataset[metric] = None  # Fill missing metrics with None

                # Reorder columns: model, dataset, then sorted metrics
                sorted_metrics = sorted(all_possible_metrics)
                df_dataset = df_dataset[["model", "dataset"] + sorted_metrics]

                # Define CSV filename based on the dataset name
                csv_filename = f"{dataset}.csv"
                csv_path = os.path.join(RESULT_DIR, csv_filename)

                # Save the dataset-specific results to CSV
                if os.path.exists(csv_path):
                    # If the CSV already exists, append the new results
                    df_existing = pd.read_csv(csv_path)
                    df_combined = pd.concat([df_existing, df_dataset], ignore_index=True)
                    df_combined.to_csv(csv_path, index=False, float_format='%.8f')
                else:
                    # If the CSV doesn't exist, create it
                    df_dataset.to_csv(csv_path, index=False, float_format='%.8f')

                print(f"Evaluation results for dataset '{dataset}' saved to {csv_path}\n")

            except Exception as e:
                print(f"Error benchmarking model {model_name} on dataset {dataset}: {e}\n")
                # Prepare a failed entry with None values for metrics
                failed_entry = {
                    "model": model_name,
                    "dataset": dataset
                }
                for metric in DATASET_METRICS.get(dataset, []):
                    failed_entry[metric] = None

                # Create a DataFrame for the failed entry
                df_failed = pd.DataFrame([failed_entry])

                # Define CSV filename based on the dataset name
                csv_filename = f"{dataset}.csv"
                csv_path = os.path.join(RESULT_DIR, csv_filename)

                # Save the failed result to CSV
                if os.path.exists(csv_path):
                    # If the CSV already exists, append the failed result
                    df_existing = pd.read_csv(csv_path)
                    df_combined = pd.concat([df_existing, df_failed], ignore_index=True)
                    df_combined.to_csv(csv_path, index=False, float_format='%.8f')
                else:
                    # If the CSV doesn't exist, create it
                    df_failed.to_csv(csv_path, index=False, float_format='%.8f')

                print(f"Failed evaluation for dataset '{dataset}' recorded in {csv_path}\n")

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# define the config_map
config_map = {
1: {"dataset": "nordjylland-news", "model": "model1"},
2: {"dataset": "nordjylland-news", "model": "model1_with_pretrain"},
3: {"dataset": "nordjylland-news", "model": "model2"},
4: {"dataset": "nordjylland-news", "model": "model2_with_pretrain"},
5: {"dataset": "nordjylland-news", "model": "model3"},
6: {"dataset": "nordjylland-news", "model": "model3_with_pretrain"},
7: {"dataset": "nordjylland-news", "model": "model4"},
8: {"dataset": "nordjylland-news", "model": "model4_with_pretrain"},
9: {"dataset": "nordjylland-news", "model": "model5"},
10: {"dataset": "nordjylland-news", "model": "model5_with_pretrain"},
11: {"dataset": "scandiqa-da", "model": "model1"},
12: {"dataset": "scandiqa-da", "model": "model1_with_pretrain"},
13: {"dataset": "scandiqa-da", "model": "model2"},
14: {"dataset": "scandiqa-da", "model": "model2_with_pretrain"},
15: {"dataset": "scandiqa-da", "model": "model3"},
16: {"dataset": "scandiqa-da", "model": "model3_with_pretrain"},
17: {"dataset": "scandiqa-da", "model": "model4"},
18: {"dataset": "scandiqa-da", "model": "model4_with_pretrain"},
19: {"dataset": "scandiqa-da", "model": "model5"},
20: {"dataset": "scandiqa-da", "model": "model5_with_pretrain"},
21: {"dataset": "scala-da", "model": "model1"},
22: {"dataset": "scala-da", "model": "model1_with_pretrain"},
23: {"dataset": "scala-da", "model": "model2"},
24: {"dataset": "scala-da", "model": "model2_with_pretrain"},
25: {"dataset": "scala-da", "model": "model3"},
26: {"dataset": "scala-da", "model": "model3_with_pretrain"},
27: {"dataset": "scala-da", "model": "model4"},
28: {"dataset": "scala-da", "model": "model4_with_pretrain"},
29: {"dataset": "scala-da", "model": "model5"},
30: {"dataset": "scala-da", "model": "model5_with_pretrain"},
}



if __name__ == "__main__":   
    # choose config number:     
    # Define model names to evaluate
    #args = get_args_eval()
    #config_num = args.config_num
    config_num = 10   
    
    model_name = config_map[config_num]["model"]
    dataset = config_map[config_num]["dataset"]
    
    MODEL_DIR = f"models_final/Instruction/{model_name}"
    RESULT_DIR = f"result/instruction/{model_name}/{dataset}"

    #MODEL_DIR = "models_final/Instruction/unsup_and_instruct_2/"
    #RESULT_DIR = "result/instruction/unsup_and_instruct_2/"

    # make the RESULT_DIR if it does not exist
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)  

    # Access the logger created by the package
    logger = logging.getLogger("scandeval")
    # create file for logging
    

    # Create a custom handler to log to a file
    file_handler = logging.FileHandler(os.path.join(RESULT_DIR, "output.log")
    , mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s ⋅ %(message)s'))

    # Add the handler to the logger
    logger.addHandler(file_handler)
    
    evaluate_final_model(
        MODEL_DIR=MODEL_DIR,
        RESULT_DIR=RESULT_DIR,
        DATASETS=[dataset],
        LANGUAGE="da",
        FRAMEWORK="pytorch",
        DEVICE="cuda",
        NUM_ITERATIONS=1, # maybe consider changing module source code since this is long prompt
        DEBUG=True)

    
    
