"""
evaluation_final_model_modified.py

Evaluates only the 'final_model' in the specified folder on the ScandiQA-DA and ScaLA-Da datasets.
Saves the results to a CSV file named 'final_benchmark.csv'.
Implements error handling to ensure smooth execution and accommodates different evaluation metrics for each dataset.
"""

import os
import pandas as pd
from scandeval import Benchmarker

# Define a mapping from dataset names to their respective metrics
DATASET_METRICS = {
    "scandiqa-da": ["test_em", "test_f1", "test_em_se", "test_f1_se"],
    "scala-da": ["test_mcc", "test_macro_f1", "test_mcc_se", "test_macro_f1_se"],
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
    CSV_FILENAME="final_benchmark.csv",
    DATASETS=["scandiqa-da", "ScaLA-Da"],
    LANGUAGE="da",
    FRAMEWORK="pytorch",
    DEVICE="cuda",
    NUM_ITERATIONS=10
):
    """
    Evaluates the 'final_model' on specified datasets and saves the results to a CSV file.
    Accommodates different evaluation metrics based on the dataset.
    """
    # Ensure result directory exists
    ensure_directory(RESULT_DIR)

    # Define path for CSV
    csv_path = os.path.join(RESULT_DIR, CSV_FILENAME)

    try:
        # Get the final model path
        final_model_path = get_final_model_path(MODEL_DIR)
        model_name = os.path.basename(final_model_path)

        print(f"\nEvaluating model: {model_name} on datasets: {DATASETS}\n")

        # Initialize the Benchmarker
        benchmark = Benchmarker(
            progress_bar=True,
            save_results=False,  # We handle saving ourselves
            device=DEVICE,
            verbose=True,
            num_iterations=NUM_ITERATIONS,
            force=True
        )

        # List to hold all results
        all_results = []

        for dataset in DATASETS:
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

                # Append the results
                all_results.append(result_entry)

            except Exception as e:
                print(f"Error benchmarking model {model_name} on dataset {dataset}: {e}\n")
                # Append a result with None values to indicate failure
                failed_entry = {
                    "model": model_name,
                    "dataset": dataset
                }
                # Set all metrics for this dataset to None
                for metric in DATASET_METRICS.get(dataset, []):
                    failed_entry[metric] = None
                all_results.append(failed_entry)

        # Create DataFrame with all results
        df_final = pd.DataFrame(all_results)

        # Ensure all possible metric columns are present in the DataFrame
        # This is useful if different datasets have different metrics
        all_possible_metrics = set()
        for metrics in DATASET_METRICS.values():
            all_possible_metrics.update(metrics)
        for metric in all_possible_metrics:
            if metric not in df_final.columns:
                df_final[metric] = None  # Fill missing metrics with None

        # Reorder columns: model, dataset, then sorted metrics
        sorted_metrics = sorted(all_possible_metrics)
        df_final = df_final[["model", "dataset"] + sorted_metrics]

        # Save the results to CSV
        df_final.to_csv(csv_path, index=False, float_format='%.8f')
        print(f"All evaluation results saved to {csv_path}")

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Constants
    timestamp = "20241213061301"
    MODEL_DIR = f"models/instruction/{timestamp}"
    RESULT_DIR = f"result/instruction/{timestamp}"

    evaluate_final_model(
        MODEL_DIR=MODEL_DIR,
        RESULT_DIR=RESULT_DIR,
        CSV_FILENAME="final_benchmark.csv",
        DATASETS=["scandiqa-da", "scala-da"],  # Ensure dataset names match exactly
        LANGUAGE="da",
        FRAMEWORK="pytorch",
        DEVICE="cuda",
        NUM_ITERATIONS=10
    )
