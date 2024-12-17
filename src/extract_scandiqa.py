import re
import pandas as pd

# Define log file paths and corresponding output CSV paths
log_file_paths = [
    'models_final/output_10_its/model2_with_pretrain/scandiqa-da/output_clean.log',
    'models_final/output_10_its/model5/scandiqa-da/output_clean.log'
]
output_csv_paths = [
    'models_final/output_10_its/model2_with_pretrain/scandiqa-da/parsed_results_model2_pretrain.csv',
    'models_final/output_10_its/model5/scandiqa-da/parsed_results_model5.csv'
]

# Function to parse a single log file and return a DataFrame
def parse_log_file(log_file_path):
    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_content = f.read()

    # Regex to capture each "Input" block, along with Prediction and Label
    pattern = re.compile(
        r"Input:\s*'(.*?)'\s*Prediction:\s*'(.*?)'\s*Label:\s*'(.*?)'",
        re.DOTALL
    )
    matches = pattern.findall(log_content)

    data = []

    for match in matches:
        full_input_block = match[0]
        prediction = match[1].strip()
        label = match[2].strip()

        # Extract shots: Tekst, Spørgsmål, Svar
        shot_pattern = re.compile(
            r"Tekst:\s*(.*?)\s*Spørgsmål:\s*(.*?)\s*Svar med maks\. 3 ord:\s*(.*?)\n",
            re.DOTALL
        )
        shots = shot_pattern.findall(full_input_block + "\n")

        if len(shots) < 5:
            continue

        # First 4 are 4Shot, 5th is final question
        four_shots = shots[:4]
        final_shot = shots[4]

        # Combine 4Shot examples
        four_shot_strs = []
        for (c, q, a) in four_shots:
            c = c.strip()
            q = q.strip()
            a = a.strip()
            four_shot_strs.append(f"Context: {c}\nQuestion: {q}\nAnswer: {a}")
        four_shot_combined = "\n\n".join(four_shot_strs)

        # Final shot details
        final_context = final_shot[0].strip()
        final_question = final_shot[1].strip()
        final_answer = label  # Use label as the correct answer

        data.append({
            "4Shot": four_shot_combined,
            "Context": final_context,
            "Question": final_question,
            "Answer": final_answer,
            "Prediction": prediction
        })

    return pd.DataFrame(data, columns=["4Shot", "Context", "Question", "Answer", "Prediction"])

# Loop through all log files and save parsed results
for log_file_path, output_csv_path in zip(log_file_paths, output_csv_paths):
    print(f"Processing log file: {log_file_path}")
    df = parse_log_file(log_file_path)
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to: {output_csv_path}")