import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import math
from scandeval import Benchmarker

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

if __name__ == "__main__":
    # Load the evaluation results CSV file
    csv_path = "result/instruction/20241211094520/evaluation_results.csv"
    plot_path = "result/instruction/20241211094520/evaluation_plot.png"
    plot_results(csv_path, plot_path)