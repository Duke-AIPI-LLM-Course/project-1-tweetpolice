#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_lm.utils import load_model

def simple_inference(model, tokenizer_dict, prompt, max_new_tokens=100):
    """
    Simplified inference function that handles input more directly.
    """
    print(f"Model type: {type(model).__name__}")
    print(f"Tokenizer type: {type(tokenizer_dict).__name__}")
    
    # Check model configuration
    if hasattr(model, 'config'):
        print(f"Model config: {model.config}")
    
    # Get vocabulary info from tokenizer dict
    vocab_size = None
    if isinstance(tokenizer_dict, dict):
        vocab_size = tokenizer_dict.get('vocab_size', None)
        if vocab_size:
            print(f"Vocabulary size: {vocab_size}")
    
    # Create a simple numerical encoding of the prompt
    # This is a fallback approach when we can't use the actual tokenizer
    char_to_idx = {c: i % 1000 + 1 for i, c in enumerate(set(prompt))}
    # Add special tokens
    char_to_idx['<s>'] = 1
    char_to_idx['</s>'] = 2
    
    # Create a simple tokenization
    tokens = [char_to_idx.get(c, 3) for c in prompt]  # Use 3 for unknown chars
    
    # Ensure we have batch dimension
    input_ids = mx.array([tokens])
    
    print(f"Input shape: {input_ids.shape}")
    print("Attempting to run inference with simplified input...")
    
    try:
        # Try to generate a fixed number of tokens
        generated = []
        current_input = input_ids
        
        for _ in range(max_new_tokens):
            # Get model output
            output = model(current_input)
            
            # Get the prediction for the next token
            next_token = mx.argmax(output[:, -1, :], axis=-1)
            
            # Add to the list of generated tokens
            generated.append(next_token.item())
            
            # Add the new token to the input for next iteration
            current_input = mx.concatenate([current_input, next_token.reshape(1, 1)], axis=1)
            
            # Stop if we generated an EOS token (approximating as token 2)
            if next_token.item() == 2:
                break
        
        print(f"Generated {len(generated)} tokens: {generated[:10]}...")
        
        # Convert back to characters as best we can
        idx_to_char = {i: c for c, i in char_to_idx.items()}
        output_chars = [idx_to_char.get(idx, '?') for idx in generated]
        output_text = ''.join(output_chars)
        
        return f"Model output (approximate): {output_text}"
    
    except Exception as e:
        print(f"Error during simplified inference: {e}")
        import traceback
        traceback.print_exc()
        
        # Try even simpler approach - just get logits for the first token
        try:
            print("Trying an even simpler approach...")
            # Just check if the model can accept input and produce any output
            simple_input = mx.array([[1, 2, 3, 4, 5]])  # Very basic input sequence
            output = model(simple_input)
            print(f"Model produced output of shape: {output.shape}")
            return f"Model can process input, but full generation failed. Output shape: {output.shape}"
        except Exception as e2:
            print(f"Even simpler approach failed: {e2}")
            return "Unable to generate with this model. It may require a specific input format or tokenizer."

def direct_inference(input_path, model_path, output_path=None):
    """
    Run inference directly on examples from a file.
    """
    # Load the model and tokenizer
    model_path = Path(model_path)
    model, tokenizer = load_model(model_path)
    
    # Examine the model architecture
    print(f"Model architecture: {type(model).__name__}")
    print(f"Tokenizer type: {type(tokenizer).__name__}")
    
    # Try to determine expected input format
    if hasattr(model, 'config'):
        print(f"Model config: {model.config}")
    elif hasattr(model, '__dict__'):
        print(f"Model attributes: {list(model.__dict__.keys())}")
    
    # Process file or direct input
    if Path(input_path).exists():
        with open(input_path, 'r') as f:
            examples = []
            for i, line in enumerate(f):
                try:
                    example = json.loads(line)
                    examples.append(example)
                except json.JSONDecodeError:
                    print(f"Could not parse line {i+1}")
        
        print(f"Loaded {len(examples)} examples from {input_path}")
        
        results = []
        for i, example in enumerate(examples):
            print(f"\nProcessing example {i+1}:")
            
            # Extract the tweet content
            tweet_text = None
            if "messages" in example and len(example["messages"]) > 0:
                user_msg = example["messages"][0]["content"]
                tweet_match = re.search(r'Analyze the following tweet[^"]*"([^"]*)"', user_msg)
                if tweet_match:
                    tweet_text = tweet_match.group(1)
            
            if not tweet_text:
                print("Could not extract tweet text from example")
                continue
            
            prompt = f"Analyze this tweet: \"{tweet_text}\""
            print(f"Prompt: {prompt}")
            
            # Run simplified inference
            response = simple_inference(model, tokenizer, prompt)
            print(f"Response: {response}")
            
            results.append({
                "tweet": tweet_text,
                "response": response
            })
        
        # Save results if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to {output_path}")
    else:
        # Treat input as direct prompt
        response = simple_inference(model, tokenizer, input_path)
        print(f"Direct inference result: {response}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    direct_inference(args.input, args.model_path, args.output)

if __name__ == "__main__":
    main()


# #!/usr/bin/env python
# import argparse
# import json
# from pathlib import Path
# import mlx.core as mx
# from mlx_lm.utils import load_model
# from transformers import AutoTokenizer
# import re

# def manual_generate(model, tokenizer_dict, prompt, max_new_tokens=512, temperature=0.1):
#     """
#     Manual text generation function that works with dictionary tokenizers.
    
#     Args:
#         model: The MLX model
#         tokenizer_dict: Tokenizer dictionary from the fine-tuned model
#         prompt: Text prompt to generate from
#         max_new_tokens: Maximum number of new tokens to generate
#         temperature: Sampling temperature (lower = more deterministic)
    
#     Returns:
#         Generated text
#     """
#     # Try to determine the model name from config
#     model_name = tokenizer_dict.get("name", None)
#     if not model_name and "model_type" in tokenizer_dict:
#         model_name = tokenizer_dict["model_type"]
    
#     # Default to Llama if we can't determine the model type
#     if not model_name:
#         print("Model type not found in tokenizer config, defaulting to Llama")
#         model_name = "meta-llama/Llama-2-7b-chat-hf"
    
#     # Load the appropriate tokenizer from Hugging Face
#     print(f"Loading tokenizer for model: {model_name}")
#     try:
#         hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
#     except Exception as e:
#         print(f"Error loading specific tokenizer: {e}")
#         print("Falling back to Llama 2 tokenizer...")
#         hf_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    
#     # Tokenize the prompt
#     inputs = hf_tokenizer(prompt, return_tensors="np")
#     input_ids = mx.array(inputs["input_ids"][0])
    
#     print(f"Input prompt: {prompt}")
#     print(f"Tokenized to {len(input_ids)} tokens")
#     print("Generating...")
    
#     # Generation settings
#     generation_settings = {
#         "max_length": len(input_ids) + max_new_tokens,
#         "eos_token_id": hf_tokenizer.eos_token_id,
#         "pad_token_id": hf_tokenizer.eos_token_id
#     }
    
#     # Simple greedy generation
#     for _ in range(max_new_tokens):
#         # Get model predictions
#         logits = model(input_ids)
        
#         # Get the next token
#         if temperature > 0:
#             # Apply temperature
#             logits = logits[-1, :] / max(temperature, 1e-7)
#             # Sample from the distribution
#             next_token = mx.random.categorical(logits.astype(mx.float32))
#         else:
#             # Greedy selection
#             next_token = mx.argmax(logits[-1, :], axis=-1)
        
#         # Add token to sequence
#         input_ids = mx.concatenate([input_ids, next_token.reshape(1)])
        
#         # Stop if we generated an EOS token
#         if next_token.item() == generation_settings["eos_token_id"]:
#             break
    
#     # Decode the generated text
#     output = hf_tokenizer.decode(input_ids.tolist(), skip_special_tokens=True)
#     return output

# def process_jsonl_file(model, tokenizer, file_path):
#     """Process a JSONL file containing examples"""
#     results = []
    
#     with open(file_path, 'r') as f:
#         for i, line in enumerate(f):
#             try:
#                 example = json.loads(line)
                
#                 # Handle both formats
#                 if "text" in example:
#                     # The text is already in Llama format
#                     text = example["text"]
#                     match = re.search(r'\[INST\](.*?)\[/INST\]', text, re.DOTALL)
#                     if match:
#                         user_prompt = match.group(1).strip()
#                     else:
#                         user_prompt = text[:200]
#                 elif "messages" in example and len(example["messages"]) >= 1:
#                     # Extract from messages format
#                     user_prompt = example["messages"][0]["content"]
#                     # Remove [INST] tags if they exist
#                     user_prompt = re.sub(r'\[INST\](.*?)\[/INST\]', r'\1', user_prompt)
#                     user_prompt = user_prompt.strip()
#                 else:
#                     print(f"Skipping example {i+1}: Missing expected format")
#                     continue
                
#                 print(f"\nProcessing example {i+1}:")
#                 prompt = f"<s>[INST] {user_prompt} [/INST]"
                
#                 try:
#                     response = manual_generate(model, tokenizer, prompt)
#                     print(f"Generated: {response}")
#                     results.append({
#                         "prompt": prompt,
#                         "response": response
#                     })
#                 except Exception as e:
#                     print(f"Error generating response: {e}")
#             except json.JSONDecodeError:
#                 print(f"Error parsing line {i+1}")
    
#     return results

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-path", type=str, required=True)
#     parser.add_argument("--input", type=str, required=True)
#     parser.add_argument("--output", type=str, default=None)
#     args = parser.parse_args()
    
#     model_path = Path(args.model_path)
#     input_path = Path(args.input)
    
#     # Load the model and tokenizer
#     model, tokenizer = load_model(model_path)
    
#     # Check if input is a file or direct text
#     if input_path.exists() and input_path.suffix == '.jsonl':
#         print(f"Processing JSONL file: {input_path}")
#         results = process_jsonl_file(model, tokenizer, input_path)
        
#         # Save results if output path is provided
#         if args.output:
#             with open(args.output, 'w') as f:
#                 json.dump(results, f, indent=2)
#             print(f"Results saved to {args.output}")
#     else:
#         # Treat input as a direct prompt
#         prompt = f"<s>[INST] {args.input} [/INST]"
#         print(f"Generating response for direct prompt")
#         try:
#             output = manual_generate(model, tokenizer, prompt)
#             print(f"Output: {output}")
#         except Exception as e:
#             print(f"Error: {e}")
#             import traceback
#             traceback.print_exc()

# if __name__ == "__main__":
#     main()