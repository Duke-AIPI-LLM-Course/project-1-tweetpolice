# Hate Speech Detection Model Fine-tuning with MLX

This repository contains scripts for fine-tuning language models on Apple Silicon Macs using MLX framework to detect hate speech and offensive language in social media content.

## Overview

The project fine-tunes a pre-trained language model (e.g., Mistral 8B) on a dataset of annotated tweets to classify content as hate speech, offensive language, or neither. The model learns to:

1. Analyze the provided text
2. Generate reasoning about the classification (Complex Chain-of-Thought)
3. Provide a final assessment with confidence levels
4. Output a label vector representing the probabilities for each category

## Requirements

- Apple Silicon Mac (M1/M2/M3)
- Python 3.8+
- MLX framework
- ~16GB RAM

## Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -U mlx-lm
pip install pandas numpy scikit-learn

# Configure Hugging Face (if using models from HF)
huggingface-cli login  # Requires access token from huggingface.co/settings/tokens
```

## Project Structure

```
├── data/
│   ├── train.jsonl
│   ├── test.jsonl
│   └── valid.jsonl
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── inference.py
├── model/
│   └── fine-tuned_model/
├── adapters/
└── README.md
```

## Usage

### 1. Data Preprocessing

Convert your hate speech dataset to the required format:

```bash
python scripts/preprocess.py --input your_dataset.json --output-dir data
```

The script expects input data in this format:

```json
[
  {
    "Question": "Analyze the following tweet for hate speech or offensive language: \"Example tweet text\"",
    "Complex_CoT": "This tweet contains... Distribution of annotations shows...",
    "Response": "This tweet contains offensive language with high confidence...",
    "LabelVector": [0.0, 1.0, 0.0]
  },
  ...
]
```

### 2. Fine-tuning

```bash
python scripts/train.py \
  --model mlx-community/Ministral-8B-Instruct-2410-4bit \
  --data-dir data \
  --batch-size 4 \
  --iters 1000 \
  --mode all
```

This script handles:
- LoRA fine-tuning
- Model fusion (merging adapters with base model)
- Quick evaluation on test examples

### 3. Inference

Run inference on new examples:

```bash
python scripts/inference.py \
  --model-path model/fine-tuned_model \
  --examples "Analyze the following tweet for hate speech or offensive language: \"Example tweet to analyze\""
```

Or batch process:

```bash
python scripts/inference.py \
  --model-path model/fine-tuned_model \
  --input-file new_examples.json \
  --output-file results.json
```

## Advanced Configuration

The training script supports various parameters:

```
--model             Base model ID or path
--data-dir          Directory containing processed data
--batch-size        Training batch size
--learning-rate     Learning rate for training
--iters             Number of training iterations
--num-layers        Number of layers to apply LoRA to
--lora-rank         Rank of LoRA matrices
--lora-alpha        Alpha parameter for LoRA
```

## Label Vector Meaning

The model outputs a 3-element label vector representing:
- Position 0: Probability of hate speech
- Position 1: Probability of offensive language
- Position 2: Probability of neither

## Acknowledgments

- MLX framework by Apple
- Hugging Face for model hosting
- Original dataset curators and annotators
