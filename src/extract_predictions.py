import re
import evaluate

# Define the path to the log file
log_file_path = "result/instruction/unsup_and_instruct_2/output.log"

# Initialize lists to store predictions and labels
predictions = []
labels = []

# Regular expressions to match predictions and labels
prediction_pattern = re.compile(r"^Prediction:\s*'(.*)'$")
label_pattern = re.compile(r"^Label:\s*'(.*)'$")

# Open the log file and process it
with open(log_file_path, "r", encoding="utf-8") as log_file:
    lines = log_file.readlines()

# Process lines while ensuring both prediction and label exist in sequence
i = 0
while i < len(lines):
    line = lines[i].strip()

    # Check if it's a Prediction line
    if line.startswith("Prediction:"):
        prediction_match = prediction_pattern.search(line)
        if prediction_match:
            prediction = prediction_match.group(1)  # Extract the prediction

            # Check the next line for the Label
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith("Label:"):
                    label_match = label_pattern.search(next_line)
                    if label_match:
                        label = label_match.group(1)  # Extract the label

                        # Add both prediction and label only if both are found
                        predictions.append(prediction)
                        labels.append(label)

        # Skip to the next pair (move 2 lines forward)
        i += 2
    else:
        # Move to the next line if no prediction is found
        i += 1


# Output the extracted predictions and labels
print("Predictions:")
print(predictions)
print("\nLabels:")
print(labels)

#Individual scores:
scores = []


# Load the ROUGE metric from Hugging Face
metric = evaluate.load("rouge")

for idx, (pred, label) in enumerate(zip(predictions, labels)):
    # Compute ROUGE-L
    results = metric.compute(predictions=[pred], references=[label])

    # Output the ROUGE-L results
    print(f"\nROUGE-L Results for pair {idx + 1}:")
    print(results["rougeL"])
    scores.append(results["rougeL"])

# Compute ROUGE-L for all predictions and labels
results = metric.compute(predictions=predictions, references=labels)

# Output the ROUGE-L results
print("\nROUGE-L Results:")
print(results["rougeL"])