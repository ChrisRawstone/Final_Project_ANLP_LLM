"""
evaluation_all_models_with_checkpoint.py

Evaluates all LLM models in the specified folder on the ScandiQA-DA dataset.
Saves the results incrementally to a CSV file and plots the evaluation metrics.
Implements checkpointing to allow resuming from the last saved state in case of interruptions.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import math
from scandeval import Benchmarker
import argparse


def get_sorted_model_paths(model_dir):
    """
    Retrieves and sorts model directories based on step numbers.
    The 'final_model' is placed at the end.
    """
    model_subdirs = [
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
    ]

    step_models = []
    final_model = None

    step_pattern = re.compile(r"step_(\d+)")

    for subdir in model_subdirs:
        match = step_pattern.match(subdir)
        if match:
            step = int(match.group(1))
            step_models.append((step, subdir))
        elif subdir.lower() == "final_model":
            final_model = subdir

    # Sort step_models by step number
    step_models_sorted = sorted(step_models, key=lambda x: x[0])

    # Extract sorted model names
    sorted_models = [subdir for _, subdir in step_models_sorted]

    # Append final_model at the end if it exists
    if final_model:
        sorted_models.append(final_model)

    # Full paths
    sorted_model_paths = [os.path.join(model_dir, model) for model in sorted_models]

    return sorted_model_paths

def ensure_directory(dir_path):
    """Ensures that the specified directory exists. Creates it if it doesn't."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

def load_existing_results(csv_path):
    """
    Loads existing evaluation results from the CSV file if it exists.
    Returns an empty DataFrame with the correct columns if the file is empty or invalid.
    """
    required_columns = {'model', 'test_em', 'test_f1', 'test_em_se', 'test_f1_se'}

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            df = pd.read_csv(csv_path, dtype={
                'model': str,
                'test_em': float,
                'test_f1': float,
                'test_em_se': float,
                'test_f1_se': float
            })
            # Verify that required columns exist
            if not required_columns.issubset(set(df.columns)):
                print(f"CSV file is missing required columns. Expected columns: {required_columns}. Starting fresh.")
                return pd.DataFrame(columns=['model', 'test_em', 'test_f1', 'test_em_se', 'test_f1_se'])
            print(f"Loaded existing results from {csv_path}.")
            return df
        except Exception as e:
            print(f"Error loading CSV file: {e}. Starting fresh.")
            return pd.DataFrame(columns=['model', 'test_em', 'test_f1', 'test_em_se', 'test_f1_se'])
    else:
        print(f"No valid CSV file found at {csv_path}. Starting fresh.")
        return pd.DataFrame(columns=['model', 'test_em', 'test_f1', 'test_em_se', 'test_f1_se'])

def save_results_incrementally(df, csv_path):
    """
    Saves the provided DataFrame to the CSV file.
    Appends rows if the file exists and is not empty, otherwise writes the header.
    """
    if not df.empty:
        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        df.to_csv(csv_path, mode='a', header=write_header, index=False, float_format='%.8f')
        print(f"Saved results to {csv_path}.")

def plot_results(csv_path, plot_path):
    """
    Plots the evaluation metrics from the CSV file.
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print("No data available to plot.")
            return

        # Extract step numbers for plotting
        def extract_step(model_name):
            match = re.match(r"step_(\d+)", model_name)
            if match:
                return int(match.group(1))
            # elif model_name.lower() == "final_model":
            #     return 999999  # Assign a high step number for final_model
            else:
                return None

        df['step'] = df['model'].apply(extract_step)
        df = df.dropna(subset=['step'])
        df = df.sort_values('step')

        # Plot the results with confidence intervals
        plt.figure(figsize=(12, 7))

        # Plot EM with error bars
        plt.errorbar(
            df['step'],
            df['test_em'],
            yerr=df['test_em_se'],
            fmt='-o',
            label='Exact Match (EM)',
            capsize=5
        )

        # Plot F1 with error bars
        plt.errorbar(
            df['step'],
            df['test_f1'],
            yerr=df['test_f1_se'],
            fmt='-s',
            label='F1 Score',
            capsize=5
        )

        plt.xlabel('Training Steps', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.title('Model Evaluation on ScandiQA-DA with Confidence Intervals', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the plot
        plt.savefig(plot_path)
        plt.close()
        print(f"Evaluation plot saved to {plot_path}")

    except Exception as e:
        print(f"Error plotting results: {e}")



def evaluate_scandeval(MODEL_DIR, RESULT_DIR, CSV_FILENAME = "evaluation_results.csv", PLOT_FILENAME = "evaluation_plot.png", DATASET = "scandiqa-da", LANGUAGE = "da", FRAMEWORK = "pytorch", DEVICE = "cuda", NUM_ITERATIONS = 5):
    # Ensure result directory exists
    ensure_directory(RESULT_DIR)

    # Define paths
    csv_path = os.path.join(RESULT_DIR, CSV_FILENAME)
    plot_path = os.path.join(RESULT_DIR, PLOT_FILENAME)

    # Load existing results
    existing_df = load_existing_results(csv_path)
    evaluated_models = set(existing_df['model'].tolist()) if not existing_df.empty else set()

    # Get sorted model paths
    sorted_model_paths = get_sorted_model_paths(MODEL_DIR)

    if not sorted_model_paths:
        print(f"No models found in {MODEL_DIR}. Exiting.")
        return

    print(f"Found {len(sorted_model_paths)} models to evaluate.")

    # List to hold new results (optional, for future use)
    new_results = []

    for model_path in sorted_model_paths:
        model_name = os.path.basename(model_path)

        if model_name in evaluated_models:
            print(f"Skipping already evaluated model: {model_name}")
            continue  # Skip already evaluated models

        print(f"\nEvaluating model: {model_name}")

        try:
            # Initialize the Benchmarker
            benchmark = Benchmarker(
                progress_bar=True,
                save_results=False,  # We handle saving ourselves
                device=DEVICE,
                verbose=True,
                num_iterations=NUM_ITERATIONS,
                force=True
            )

            # Run the benchmark and capture the results
            results = benchmark(
                model=model_path,
                dataset=DATASET,
                language=LANGUAGE,
                framework=FRAMEWORK,
                trust_remote_code=True,
                verbose=True,
            )

            # Debugging: Print the type and contents of results
            print(f"Type of results: {type(results)}")
            print(f"Contents of results: {results}")

            # Initialize variables
            test_em_mean = None
            test_f1_mean = None
            test_em_se = None
            test_f1_se = None

            # Process results
            if isinstance(results, list):
                # Assuming 'results' is a list of BenchmarkResult objects
                # Extract 'total' from the last result to get aggregated metrics
                last_result = results[-1]
                if hasattr(last_result, 'results') and isinstance(last_result.results, dict):
                    total = last_result.results.get('total', {})
                    test_em_mean = total.get('test_em', None)
                    test_f1_mean = total.get('test_f1', None)
                    test_em_se = total.get('test_em_se', None)
                    test_f1_se = total.get('test_f1_se', None)
                else:
                    print(f"Unexpected result format for model {model_name}.")
            elif hasattr(results, 'results'):
                # Single BenchmarkResult object
                total = results.results.get('total', {})
                test_em_mean = total.get('test_em', None)
                test_f1_mean = total.get('test_f1', None)
                test_em_se = total.get('test_em_se', None)
                test_f1_se = total.get('test_f1_se', None)
            else:
                print(f"Unexpected result format for model {model_name}.")

            # Debugging: Print extracted metrics
            print(f"Extracted metrics for {model_name}: EM={test_em_mean} ± {test_em_se}, F1={test_f1_mean} ± {test_f1_se}")

            # Create DataFrame with the results
            df_new = pd.DataFrame([{
                "model": model_name,
                "test_em": test_em_mean,
                "test_f1": test_f1_mean,
                "test_em_se": test_em_se,
                "test_f1_se": test_f1_se
            }])

            # Save results incrementally
            save_results_incrementally(df_new, csv_path)

            # Append to new_results for plotting later (optional)
            new_results.append({
                "model": model_name,
                "test_em": test_em_mean,
                "test_f1": test_f1_mean,
                "test_em_se": test_em_se,
                "test_f1_se": test_f1_se
            })

            print(f"Completed evaluation for {model_name}: EM={test_em_mean} ± {test_em_se}, F1={test_f1_mean} ± {test_f1_se}")

        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            # Append a result with None values to indicate failure
            df_failed = pd.DataFrame([{
                "model": model_name,
                "test_em": None,
                "test_f1": None,
                "test_em_se": None,
                "test_f1_se": None
            }])
            save_results_incrementally(df_failed, csv_path)
            continue  # Proceed to the next model

    print("All evaluations completed.")
    # After all evaluations, plot the results
    plot_results(csv_path, plot_path)




if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Evaluate Scandeval model and save results.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models_final/Instruction/best_model_pretrain_christian",
        help="Path to the model directory."
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="models_final/Instruction/best_model_pretrain_christian",
        help="Path to save the evaluation results."
    )

    # Parse arguments
    args = parser.parse_args()

    # Constants
    MODEL_DIR = args.model_dir
    RESULT_DIR = args.result_dir

    # Call the evaluation function
    evaluate_scandeval(MODEL_DIR, RESULT_DIR)

