import os
import glob

def cut_file(input_file, max_chars=110000000):
    # Skip if file already ends with '_cut.danish'
    if input_file.endswith('_cut.danish'):
        return

    # Process file in chunks instead of reading all at once
    chunk_size = 1024 * 1024  # 1MB chunks
    output_content = ''
    chars_read = 0
    
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_cut{ext}"
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        while chars_read < max_chars:
            chunk = fin.read(chunk_size)
            if not chunk:
                break
            
            if (chars_read + len(chunk)) > max_chars:
                chunk = chunk[:max_chars - chars_read]
            
            fout.write(chunk)
            chars_read += len(chunk)
            
    print(f"Processed {input_file} -> {output_file}")

def process_directories():
    # Get absolute path to data/raw/unsupervised directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'data', 'raw', 'unsupervised')

    # Check if directory exists
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found!")
        print(f"Looking for: {base_dir}")  # Debug print
        return
        
    # Process each subdirectory
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        
        # Skip if not a directory
        if not os.path.isdir(subdir_path):
            continue
            
        # Process all txt files in subdirectory
        for txt_file in glob.glob(os.path.join(subdir_path, '*.danish')):
            cut_file(txt_file)

if __name__ == "__main__":
    print("Starting to process directories...")
    process_directories()
