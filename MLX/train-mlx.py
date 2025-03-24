#!/usr/bin/env python
# Fine-tune an LLM for hate speech detection using MLX and LoRA

import os
import argparse
import json
from pathlib import Path

def train(args):
    """Run the MLX LoRA fine-tuning"""
    
    # Print some info about the training setup
    print("\n=== Hate Speech Detection Fine-tuning ===")
    print(f"Model: {args.model}")
    print(f"Data directory: {args.data_dir}")
    print(f"Adapter path: {args.adapter_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Iterations: {args.iters}")
    print(f"Number of LoRA layers: {args.num_layers}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"LoRA alpha: {args.lora_alpha}")
    print("=======================================\n")

    # Build the mlx_lm.lora command
    cmd = [
        "python -m mlx_lm.lora",
        f"--model {args.model}",
        f"--data {args.data_dir}",
        "--train",
        "--fine-tune-type lora",
        f"--batch-size {args.batch_size}",
        f"--num-layers {args.num_layers}",
        f"--iters {args.iters}",
        f"--adapter-path {args.adapter_path}",
        f"--learning-rate {args.learning_rate}"
    ]
    
    if args.gradient_accumulation:
        cmd.append(f"--gradient-accumulation {args.gradient_accumulation}")
    
    if args.model_path:
        cmd.append(f"--model-path {args.model_path}")
        
    if args.save_every:
        cmd.append(f"--save-every {args.save_every}")
    
    # Execute the command
    command = " \\\n    ".join(cmd)
    print(f"Executing command:\n{command}\n")
    os.system(command)
    
    # Print completion message
    print("\n=== Fine-tuning Complete ===")
    print(f"Adapter saved to: {args.adapter_path}")
    print("To fuse the model with adapters, run:")
    print(f"python -m mlx_lm.fuse \\\n    --model {args.model} \\\n    --adapter-path {args.adapter_path} \\\n    --save-path {args.output_dir}/fine-tuned_model")


def fuse_model(args):
    """Fuse the base model with the trained adapter"""
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n=== Fusing Model with Adapters ===")
    print(f"Model: {args.model}")
    print(f"Adapter path: {args.adapter_path}")
    print(f"Output path: {args.output_dir}/fine-tuned_model")
    print("====================================\n")
    
    # Build the mlx_lm.fuse command
    cmd = [
        "python -m mlx_lm.fuse",
        f"--model {args.model}",
        f"--adapter-path {args.adapter_path}",
        f"--save-path {args.output_dir}/fine-tuned_model"
    ]
    
    if args.model_path:
        cmd.append(f"--model-path {args.model_path}")
    
    # Execute the command
    command = " \\\n    ".join(cmd)
    print(f"Executing command:\n{command}\n")
    os.system(command)
    
    # Print completion message
    print("\n=== Model Fusion Complete ===")
    print(f"Fine-tuned model saved to: {args.output_dir}/fine-tuned_model")


def evaluate(args):
    """Evaluate the fine-tuned model on test data"""
    import json
    import os
    from pathlib import Path
    
    # Convert string path to Path object
    model_path = Path(args.output_dir) / "fine-tuned_model"
    
    print("\n=== Evaluating Model ===")
    print(f"Model path: {model_path}")
    print(f"Test file: {args.data_dir}/test.jsonl")
    print("========================\n")
    
    # Load the test data first
    test_examples = []
    try:
        with open(f"{args.data_dir}/test.jsonl", "r") as f:
            for line in f:
                test_examples.append(json.loads(line))
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Use MLX's generate.py script with the correct parameters
    try:
        num_examples = min(5, len(test_examples))
        for i in range(num_examples):
            example = test_examples[i]
            user_message = example["messages"][0]["content"]
            ground_truth = example["messages"][1]["content"]
            
            print(f"\nTest Example {i+1}:")
            print(f"User: {user_message}")
            print(f"Ground Truth: {ground_truth}")
            
            # Create a temporary prompt file
            prompt_file = f"temp_prompt_{i}.txt"
            with open(prompt_file, "w") as f:
                f.write(f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n")
            
            # Use the mlx_lm.generate command line interface with correct parameters
            # The error shows we need to use --model instead of --model-path
            output_file = f"temp_output_{i}.txt"
            cmd = [
                "python -m mlx_lm.generate",
                f"--model {model_path}",  # Changed from --model-path to --model
                f"--prompt {prompt_file}",  # Changed from --prompt-file to --prompt
                "--max-tokens 512",
                "--temp 0.0",
                f"> {output_file}"
            ]
            
            command = " ".join(cmd)
            print("Generating response...")
            os.system(command)
            
            # Read the generated output
            try:
                with open(output_file, "r") as f:
                    generated = f.read().strip()
                print(f"Generated: {generated}")
            except Exception as e:
                print(f"Failed to read generated output: {e}")
            
            # Clean up temporary files
            try:
                if os.path.exists(prompt_file):
                    os.remove(prompt_file)
                if os.path.exists(output_file):
                    os.remove(output_file)
            except Exception as e:
                print(f"Failed to clean up temporary files: {e}")
            
            print("-" * 50)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for hate speech detection with MLX")
    
    # Common arguments
    parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-1B-Instruct-4bit",
                        help="Model to fine-tune (Hugging Face ID or path)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to local model weights (if not using Hugging Face)")
    parser.add_argument("--data-dir", type=str, default="/Users/ruoxinwang/Desktop/Duke/Large_Language_Models/TweetPlice/project-1-tweetpolice/MLX/llama_data",
                        help="Directory containing the processed data files")
    parser.add_argument("--adapter-path", type=str, default="adapters",
                        help="Path to save/load the LoRA adapters")
    parser.add_argument("--output-dir", type=str, default="model",
                        help="Directory to save the fine-tuned model")
    
    # Mode selection (train, fuse, evaluate, all)
    parser.add_argument("--mode", type=str, choices=["train", "fuse", "evaluate", "all"],
                        default="all", help="Operation mode")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=5e-6,
                        help="Learning rate for training")
    parser.add_argument("--iters", type=int, default=200,
                        help="Number of training iterations")
    parser.add_argument("--num-layers", type=int, default=16,
                        help="Number of layers to apply LoRA to")
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="Rank of LoRA matrices")
    parser.add_argument("--lora-alpha", type=float, default=16,
                        help="Alpha parameter for LoRA")
    parser.add_argument("--gradient-accumulation", type=int, default=None,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--save-every", type=int, default=None,
                        help="Save checkpoint every N iterations")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.adapter_path, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the selected mode(s)
    if args.mode in ["train", "all"]:
        train(args)
    
    if args.mode in ["fuse", "all"]:
        fuse_model(args)
    
    if args.mode in ["evaluate", "all"]:
        evaluate(args)

if __name__ == "__main__":
    main()
