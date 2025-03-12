#!/usr/bin/env python3
"""
Example script demonstrating how to use datasets and models together.
"""
import os
import sys
import json
import fire

# Add the parent directory to the path so we can import the lot package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the lot package
from lot.datasets import AQuA, MMLU, StrategyQA, CommonsenseQA
from lot.models import opensource_API_models, ModelAdapter

def load_dataset(dataset_name, data_dir):
    """Load a dataset by name."""
    if dataset_name == "aqua":
        return AQuA(os.path.join(data_dir, "aqua.jsonl"))
    elif dataset_name == "mmlu":
        return MMLU(os.path.join(data_dir, "mmlu.json"))
    elif dataset_name == "strategyqa":
        return StrategyQA(os.path.join(data_dir, "strategyqa.json"))
    elif dataset_name == "commonsenseqa":
        return CommonsenseQA(os.path.join(data_dir, "commonsenseqa.jsonl"))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def print_example(example):
    """Print a formatted example from a dataset."""
    for key, value in example.items():
        if isinstance(value, list) and len(value) > 3:
            print(f"{key}: [{value[0]}, {value[1]}, ...]")
        else:
            print(f"{key}: {value}")

def save_example(example, output_path):
    """Save an example to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(example, f, indent=2, ensure_ascii=False)
    print(f"Saved example to {output_path}")

def main(dataset="aqua", sample=1):
    """
    Demonstrate dataset and model usage.
    
    Args:
        dataset: Dataset to use. Options: aqua, mmlu, strategyqa, commonsenseqa.
        sample: Number of examples to sample.
    """
    # Define the data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    # Load the selected dataset
    dataset_obj = load_dataset(dataset, data_dir)
    
    # Print dataset info
    print(f"Dataset: {dataset}")
    print(f"Size: {len(dataset_obj)}")
    
    # Sample examples
    for i in range(min(sample, len(dataset_obj))):
        print(f"\nExample {i+1}:")
        example = dataset_obj[i]
        print_example(example)

        # Initialize model with adapter
        base_model = opensource_API_models(
            model="meta-llama/Meta-Llama-3-8B-Instruct-Lite", 
            max_tokens=1000,
            system_prompt_type="default"
        )
        model = ModelAdapter(base_model)

        # Format prompt from example
        prompt = f"Question: {example['question']}\n"   

        # Get model response
        print("\nModel response:")
        response = model.generate(prompt, temperature=0.7)
        print(response.text[0])

if __name__ == "__main__":
    fire.Fire(main) 