import pandas as pd
import numpy as np
from pathlib import Path

def process_text_file(file_path):
    """Process a large text file in chunks to minimize memory usage"""
    print(f"\nProcessing: {file_path}")
    
    total_chars = 0
    total_lines = 0
    line_lengths = []
    chunk_size = 100000
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    # First find working encoding
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.readline()  # Try reading one line
            break
        except UnicodeDecodeError:
            if encoding == encodings[-1]:
                raise Exception(f"Could not decode file with any of the following encodings: {encodings}")
            continue
    
    # Process file in chunks
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            while True:
                chunk = f.readlines(chunk_size)
                if not chunk:
                    break
                    
                # Process chunk
                for line in chunk:
                    line_len = len(line)
                    total_chars += line_len
                    total_lines += 1
                    line_lengths.append(line_len)
                
                # Print progress for large files
                if total_lines % 1000000 == 0:
                    print(f"Processed {total_lines:,} lines...")
                    
    except Exception as e:
        print(f"Error while processing file: {str(e)}")
        return None
        
    # Calculate statistics using numpy for efficiency
    line_lengths = np.array(line_lengths)
    stats = {
        'file_name': file_path.name,
        'total_characters': total_chars,
        'total_lines': total_lines,
        'num_sentences': total_lines,
        'avg_sentence_length': np.mean(line_lengths),
        'max_sentence_length': np.max(line_lengths),
        'min_sentence_length': np.min(line_lengths)
    }
    
    # Print summary
    print("\nFile Processing Complete!")
    print(f"Total characters: {stats['total_characters']:,}")
    print(f"Total lines: {stats['total_lines']:,}")
    print(f"Average line length: {stats['avg_sentence_length']:.2f} characters")
    print(f"Longest line: {stats['max_sentence_length']} characters")
    print(f"Shortest line: {stats['min_sentence_length']} characters")
    
    return pd.DataFrame([stats])

# Get all directories in the unsupervised folder
unsupervised_path = Path('data/raw/unsupervised')
if not unsupervised_path.exists():
    raise FileNotFoundError(f"Could not find directory: {unsupervised_path}")

# Create an empty list to store all statistics
all_stats = []

# Process each directory
for lang_dir in unsupervised_path.iterdir():
    if lang_dir.is_dir():
        print(f"\nProcessing directory: {lang_dir.name}")
        
        # Process all text files in the directory
        for text_file in lang_dir.glob('*.txt*'):
            try:
                stats_df = process_text_file(text_file)
                stats_df['Name'] = lang_dir.name
                all_stats.append(stats_df)
                
            except Exception as e:
                print(f"Error processing {text_file}: {str(e)}")

# Combine all statistics and save to a single CSV
if all_stats:
    final_stats = pd.concat(all_stats, ignore_index=True)
    output_path = Path('data/processed/text_data_statistics.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_stats.to_csv(output_path, index=False)
    print(f"\nStatistics saved to '{output_path}'")
