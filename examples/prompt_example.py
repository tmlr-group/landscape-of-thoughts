"""
Example script showing how to use the dataset-specific prompts.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lot.datasets import load_dataset, get_prompt, get_answer_pattern, DATASET_PROMPTS

def main():
    """
    Main function demonstrating the usage of dataset-specific prompts.
    """
    # List all available prompts
    print("Available dataset prompts:")
    for dataset_name, prompts in DATASET_PROMPTS.items():
        print(f"  - {dataset_name}:")
        for method, prompt in prompts.items():
            print(f"    - {method}: {prompt[:50]}...")
    print()
    
    # Get a specific prompt
    dataset_name = "aqua"
    method = "cot"
    prompt = get_prompt(dataset_name, method)
    print(f"Prompt for {dataset_name} with {method} method:")
    print(prompt)
    print()
    
    # Get answer pattern for a dataset
    pattern = get_answer_pattern(dataset_name)
    print(f"Answer pattern for {dataset_name}: {pattern}")
    print()
    
    # Load a dataset and access its prompts
    data_path = "data/aqua.jsonl"
    if os.path.exists(data_path):
        dataset = load_dataset(dataset_name, data_path)
        
        # Get prompts from the dataset
        print(f"Prompts available in the dataset:")
        for method, prompt in dataset.prompts.items():
            print(f"  - {method}: {prompt[:50]}...")
        print()
        
        # Format a prompt for a specific example
        if len(dataset) > 0:
            idx = 0
            formatted_prompt = dataset.format_prompt(idx, "cot")
            print(f"Formatted prompt for example {idx} with 'cot' method:")
            print(formatted_prompt)
            print()
            
            # Get the query for the example
            query = dataset.get_query(idx)
            print(f"Query for example {idx}:")
            print(query)
    else:
        print(f"Dataset file does not exist: {data_path}")

if __name__ == "__main__":
    main() 