import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_data(input_file, output_dir="data", test_size=0.1, valid_size=0.1, seed=42):
    """
    Preprocess hate speech data for fine-tuning.
    
    Args:
        input_file (str): Path to input JSON file containing hate speech data
        output_dir (str): Directory to save processed data
        test_size (float): Proportion of data to use for testing
        valid_size (float): Proportion of data to use for validation
        seed (int): Random seed for reproducibility
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {input_file}...")
    
    # Load data based on file extension
    if input_file.endswith('.json'):
        with open(input_file, 'r') as f:
            data = json.load(f)
    elif input_file.endswith('.jsonl'):
        data = []
        with open(input_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError("Input file must be JSON or JSONL format")
    
    print(f"Loaded {len(data)} examples")
    
    # Prepare data for MLX fine-tuning format
    processed_data = []
    
    for item in data:
        # Format data for instruction fine-tuning
        processed_item = {
            "messages": [
                {"role": "user", "content": item["Question"]},
                {"role": "assistant", "content": f"I'll analyze this tweet for hate speech or offensive language.\n\n{item['Complex_CoT']}\n\n{item['Response']}"}
            ]
        }
        processed_data.append(processed_item)
    
    # Split data into train, validation, and test sets
    train_data, test_data = train_test_split(processed_data, test_size=test_size, random_state=seed)
    train_data, valid_data = train_test_split(train_data, test_size=valid_size/(1-test_size), random_state=seed)
    
    print(f"Split data into {len(train_data)} train, {len(valid_data)} validation, and {len(test_data)} test examples")
    
    # Save processed data
    with open(os.path.join(output_dir, "train.jsonl"), 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
            
    with open(os.path.join(output_dir, "valid.jsonl"), 'w') as f:
        for item in valid_data:
            f.write(json.dumps(item) + '\n')
            
    with open(os.path.join(output_dir, "test.jsonl"), 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved processed data to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess hate speech data for fine-tuning")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON/JSONL file")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory to save processed data")
    parser.add_argument("--test-size", type=float, default=0.1, help="Proportion of data to use for testing")
    parser.add_argument("--valid-size", type=float, default=0.1, help="Proportion of data to use for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    preprocess_data(
        input_file=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        valid_size=args.valid_size,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
