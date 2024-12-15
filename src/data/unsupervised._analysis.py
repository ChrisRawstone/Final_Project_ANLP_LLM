import pandas as pd
import numpy as np
from pathlib import Path

def process_text_file(file_path):
    """Process a large text file in chunks to minimize memory usage"""
    print(f"\nProcessing: {file_path}")
    
    total_tokens = 0
    total_lines = 0
    tokens_per_line = []
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
                    # Split line into tokens (simple whitespace tokenization)
                    tokens = line.split()
                    num_tokens = len(tokens)
                    total_tokens += num_tokens
                    total_lines += 1
                    tokens_per_line.append(num_tokens)
                
                # Print progress for large files
                if total_lines % 1000000 == 0:
                    print(f"Processed {total_lines:,} lines...")
                    
    except Exception as e:
        print(f"Error while processing file: {str(e)}")
        return None
        
    # Calculate statistics using numpy for efficiency
    tokens_per_line = np.array(tokens_per_line)
    stats = {
        'file_name': file_path.name,
        'total_tokens': total_tokens,
        'total_lines': total_lines,
        'num_sentences': total_lines,
        'avg_tokens_per_sentence': np.mean(tokens_per_line),
        'max_tokens_per_sentence': np.max(tokens_per_line),
        'min_tokens_per_sentence': np.min(tokens_per_line)
    }
    
    # Print summary
    print("\nFile Processing Complete!")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Total lines: {stats['total_lines']:,}")
    print(f"Average tokens per line: {stats['avg_tokens_per_sentence']:.2f}")
    print(f"Most tokens in a line: {stats['max_tokens_per_sentence']}")
    print(f"Fewest tokens in a line: {stats['min_tokens_per_sentence']}")
    
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
