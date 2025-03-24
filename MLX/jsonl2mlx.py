import json
import os
from pathlib import Path
import argparse

def convert_to_mlx_format(input_file, output_dir):
    """
    Convert hate speech data to the format MLX expects for fine-tuning.
    
    Args:
        input_file (str): Path to input JSON or JSONL file
        output_dir (str): Directory to save JSONL files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine if the input is already JSONL format
    input_path = Path(input_file)
    is_jsonl = False
    
    # Check the file content to determine format
    with open(input_path, 'r') as f:
        first_line = f.readline().strip()
        try:
            # Try to parse as JSON
            json.loads(first_line)
            # If we get here and there's a second line, it's likely JSONL
            second_line = f.readline().strip()
            is_jsonl = bool(second_line)
        except json.JSONDecodeError:
            # Not valid JSON on first line, might be a full JSON file
            is_jsonl = False
    
    # Read the data
    data = []
    if is_jsonl:
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
    else:
        with open(input_path, 'r') as f:
            try:
                file_content = json.load(f)
                if isinstance(file_content, list):
                    data = file_content
                else:
                    data = [file_content]  # Single object
            except json.JSONDecodeError:
                print(f"Error: {input_file} is not valid JSON.")
                return
    
    # Check if we have data
    if not data:
        print(f"No data found in {input_file}")
        return
    
    # Split data into train, valid, test sets (80%, 10%, 10%)
    total = len(data)
    train_count = int(total * 0.8)
    valid_count = int(total * 0.1)
    
    train_data = data[:train_count]
    valid_data = data[train_count:train_count + valid_count]
    test_data = data[train_count + valid_count:]
    
    # Ensure each split has at least one example
    if not train_data:
        train_data = [data[0]]
    if not valid_data:
        valid_data = [data[0]]
    if not test_data:
        test_data = [data[0]]
    
    # Convert to MLX format and write to files
    for split_name, split_data in [("train", train_data), ("valid", valid_data), ("test", test_data)]:
        output_file = output_path / f"{split_name}.jsonl"
        
        with open(output_file, 'w') as f:
            for item in split_data:
                # Format data for MLX fine-tuning (preserve original format inside message content)
                mlx_item = {
                    "messages": [
                        {"role": "user", "content": item["Question"]},
                        {"role": "assistant", "content": f"I'll analyze this tweet for hate speech or offensive language.\n\n{item['Complex_CoT']}\n\n{item['Response']}"}
                    ]
                }
                f.write(json.dumps(mlx_item) + '\n')
        
        print(f"Saved {len(split_data)} examples to {output_file}")
    
    print(f"\nConversion complete. MLX-compatible JSONL files are in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert hate speech data to MLX format for fine-tuning")
    parser.add_argument("--input-file", type=str, required=True, help="Path to input JSON or JSONL file")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory to save JSONL files")
    
    args = parser.parse_args()
    
    convert_to_mlx_format(args.input_file, args.output_dir)

if __name__ == "__main__":
    main()