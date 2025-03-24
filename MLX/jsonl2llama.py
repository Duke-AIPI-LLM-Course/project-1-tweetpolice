import json
import os
from pathlib import Path
import argparse

def convert_to_llama_mlx_format(input_file, output_dir):
    """
    Convert dataset from 'messages' format to Llama format compatible with MLX-LM.
    
    Args:
        input_file (str): Path to input JSONL file
        output_dir (str): Directory to save the converted JSONL files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Process each line
    llama_formatted_data = []
    for line in lines:
        if not line.strip():  # Skip empty lines
            continue
        
        try:
            data = json.loads(line)
            
            # Extract user message and assistant response
            if "messages" in data and len(data["messages"]) >= 2:
                user_message = data["messages"][0]["content"]
                assistant_response = data["messages"][1]["content"]
                
                # Format for Llama with MLX-LM conventions
                llama_format = {
                    "messages": [
                        {"role": "user", "content": f"[INST] {user_message} [/INST]"},
                        {"role": "assistant", "content": assistant_response}
                    ]
                }
                
                llama_formatted_data.append(llama_format)
            else:
                print(f"Warning: Skipping line with invalid format: {line[:50]}...")
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON: {line[:50]}...")
    
    # Split into train, validation, and test sets (80%, 10%, 10%)
    num_examples = len(llama_formatted_data)
    train_size = int(num_examples * 0.8)
    val_size = int(num_examples * 0.1)
    
    train_data = llama_formatted_data[:train_size]
    val_data = llama_formatted_data[train_size:train_size + val_size]
    test_data = llama_formatted_data[train_size + val_size:]
    
    # Save the converted data
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        output_file = output_path / f"{split_name}.jsonl"
        
        with open(output_file, 'w') as f:
            for item in split_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Saved {len(split_data)} examples to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert dataset to Llama instruction format for MLX-LM")
    parser.add_argument("--input-file", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output-dir", type=str, default="llama_data", help="Directory to save converted files")
    
    args = parser.parse_args()
    convert_to_llama_mlx_format(args.input_file, args.output_dir)
    print(f"Conversion complete. Llama-formatted data saved to {args.output_dir}")

if __name__ == "__main__":
    main()