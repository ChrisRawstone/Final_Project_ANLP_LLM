import re
import json
import pandas as pd


# def parse_log(file_path):
#     # Regex pattern to capture Input, Prediction, and Label with adjusted Input field
#     pattern = re.compile(
#         r"Nyhedsartikel:\s*(.*?)\s*Resumé:"  # Capture everything between 'Nyhedsartikel:' and 'Resumé:'
#         r".*?"  # Match the remaining content (Prediction and Label) without affecting the Input
#         r"Prediction:\s*'(.*?)'\s*"
#         r"Label:\s*'(.*?)'",
#         re.DOTALL  # Allows matching across multiple lines
#     )

def parse_log(file_path):
    # Regex pattern to capture two sets of Nyhedsartikel, Resumé, Prediction, and Label
    pattern = re.compile(
        r"Nyhedsartikel:\s*(.*?)\s*Resumé:\s*(.*?)\s*"  # First Nyhedsartikel and Resumé
        r"Nyhedsartikel:\s*(.*?)\s*Resumé:\s*(.*?)\s*"  # Second Nyhedsartikel and Resumé
        r"Prediction:\s*'(.*?)'\s*"  # Prediction
        r"Label:\s*'(.*?)'",  # Label
        re.DOTALL  # Allows matching across multiple lines
    )

    results = []

    # Read the log file
    with open(file_path, 'r', encoding='utf-8') as log_file:
        log_content = log_file.read()

        # Find all matches
        matches = pattern.finditer(log_content)
        for match in matches:
            # Extracting parts from the match
            nyhedsartikel_content_1 = match.group(1).strip()  # First Nyhedsartikel
            resumé_content_1 = match.group(2).strip()  # First Resumé
            nyhedsartikel_content_2 = match.group(3).strip()  # Second Nyhedsartikel
            resumé_content_2 = match.group(4).strip()  # Second Resumé
            prediction = match.group(5).strip()  # Prediction
            label = match.group(6).strip()  # Label

            # Format the results with both Nyhedsartikel sections
            results.append({
                "Fewshot_artikel": nyhedsartikel_content_1,
                "Fewshot_resume": resumé_content_1,
                "Input": nyhedsartikel_content_2,
                "Dummy": resumé_content_2,
                "Prediction": prediction,
                "Label": label
            })

    return results

def save_to_file(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, indent=4, ensure_ascii=False)  # Save as JSON with proper formatting


if __name__ == "__main__":
    model_names = ["model2_with_pretrain", "model5"]
    for model_name in model_names:
        log_file_path = f"models_final/output_10_its/{model_name}/nordjylland-news/output_clean.log"
        output_file_path_df = f"models_final/output_10_its/{model_name}/nordjylland-news/df.json"
        output_file_path_index_to_input = f"models_final/output_10_its/{model_name}/nordjylland-news/index_to_input.json"
        parsed_entries = parse_log(log_file_path)

        # Debug: Check if entries were found
        if not parsed_entries:
            print("No matches found. Please check the log file format or adjust the regex.")

        # turn into a dataframe
        df = pd.DataFrame(parsed_entries)
        # add an idx to rows that share the same Fewshot_artikel
        unique_fewshots_artikel = df['Fewshot_artikel'].unique()

        # save a mapping from index to input
        index_to_input = {i: input for i, input in enumerate(unique_fewshots_artikel)}
        # save the mapping
        with open(output_file_path_index_to_input, 'w', encoding='utf-8') as f:
            json.dump(index_to_input, f)

        # create a new column that is the index of the input
        df['Iteration'] = df['Fewshot_artikel'].apply(lambda x: list(unique_fewshots_artikel).index(x))
        # drop the dummy column
        df = df.drop(columns=['Dummy'])
        # save the dataframe
        df.to_json(output_file_path_df, orient='records', lines=True)




