from scandeval import Benchmarker

# Initialize the Benchmarker with desired default settings
benchmark = Benchmarker(
    progress_bar=True,             # Show progress bars
    save_results=True,             # Save results to 'scandeval_benchmark_results.jsonl'
    device="cuda",                 # Use GPU for evaluation
    verbose=True,                   # Enable verbose logging
    num_iterations=3,              # Number of iterations to run the benchmark
)

# Run the benchmark on your local model
results = benchmark(
    model="models/long_train_hpc/step_2600",  # Path to your local model
    dataset="scandiqa-da",
    language="da",  # Specify your language
    device="cuda",  # Optional: Specify device (e.g., 'cuda' for GPU)
    verbose=True  # Optional: Enable verbose output
)

# Print the benchmark results
for result in results:
    print(result)

