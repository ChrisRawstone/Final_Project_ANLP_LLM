"""
plot_evaluation.py

Loads the evaluation_results.csv file and plots the evaluation metrics (Exact Match and F1 Score)
with their corresponding standard errors. The plot is saved as evaluation_plot.png in the result directory.

Usage:
    python plot_evaluation.py --max_step 4000

Parameters:
    --max_step: (Optional) Maximum training step to include in the plot. Default is 4000.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Constants
RESULT_DIR = "result"
CSV_FILENAME = "evaluation_results.csv"
PLOT_FILENAME = "evaluation_plot.png"

def ensure_directory(dir_path):
    """
    Ensures that the specified directory exists. Creates it if it doesn't.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

def load_csv_file(csv_path):
    """
    Loads the CSV file and validates the presence of required columns.
    Returns a DataFrame if successful, otherwise exits the script.
    """
    required_columns = {'model', 'test_em', 'test_f1', 'test_em_se', 'test_f1_se'}
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}. Please ensure the file exists.")
        exit(1)
    
    try:
        df = pd.read_csv(csv_path)
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            print(f"Error: CSV file is missing required columns: {missing_columns}")
            exit(1)
        if df.empty:
            print(f"Error: CSV file at {csv_path} is empty.")
            exit(1)
        print(f"Successfully loaded CSV file with {len(df)} entries.")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        exit(1)

def extract_step_number(model_name):
    """
    Extracts the numerical step number from the model name.
    For example, 'step_1300' returns 1300.
    If 'final_model', returns a very high number to place it at the end.
    Returns None if the pattern doesn't match.
    """
    match = re.match(r"step_(\d+)", model_name.lower())
    if match:
        return int(match.group(1))
    # elif model_name.lower() == "final_model":
    #     return 999999  # Assign a high step number for final_model
    else:
        return None

def plot_results(df, plot_path, max_step):
    """
    Plots the Exact Match (EM) and F1 Score with error bars.
    Saves the plot to the specified path.
    Only includes models with step numbers less than or equal to max_step.
    """
    # Extract step numbers
    df['step'] = df['model'].apply(extract_step_number)
    
    # Drop entries where step is None
    initial_count = len(df)
    df = df.dropna(subset=['step'])
    if len(df) < initial_count:
        print(f"Warning: Dropped {initial_count - len(df)} entries due to unrecognized model names.")
    
    # Convert step to integer
    df['step'] = df['step'].astype(int)
    
    # Filter based on max_step
    if max_step is not None:
        df = df[df['step'] <= max_step]
        print(f"Filtered models to include only those with step <= {max_step}. Remaining models: {len(df)}")
    
    # Sort by step number
    df = df.sort_values('step')
    
    if df.empty:
        print("Error: No valid data to plot after applying step filter.")
        return
    
    # Plotting
    plt.figure(figsize=(12, 7))
    
    # Plot Exact Match (EM)
    plt.errorbar(
        df['step'],
        df['test_em'],
        yerr=df['test_em_se'],
        fmt='-o',
        label='Exact Match (EM)',
        capsize=5,
        markersize=5,
        linewidth=1.5
    )
    
    # Plot F1 Score
    plt.errorbar(
        df['step'],
        df['test_f1'],
        yerr=df['test_f1_se'],
        fmt='-s',
        label='F1 Score',
        capsize=5,
        markersize=5,
        linewidth=1.5
    )
    
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Score (%)', fontsize=14)
    plt.title('Model Evaluation on ScandiQA-DA with Confidence Intervals', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(plot_path)
    plt.close()
    print(f"Evaluation plot saved to {plot_path}")

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Plot evaluation metrics from CSV file.")
    parser.add_argument(
        '--max_step',
        type=int,
        default=13000,
        help='Maximum training step to include in the plot. Default is 4000.'
    )
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    max_step = args.max_step
    
    # Ensure result directory exists
    ensure_directory(RESULT_DIR)
    
    # Define paths
    csv_path = os.path.join(RESULT_DIR, CSV_FILENAME)
    plot_path = os.path.join(RESULT_DIR, PLOT_FILENAME)
    
    # Load CSV data
    df = load_csv_file(csv_path)
    
    # Plot the results with the specified max_step
    plot_results(df, plot_path, max_step)

if __name__ == "__main__":
    main()
