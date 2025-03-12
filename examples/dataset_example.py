"""
Example script showing how to use the new dataset interface.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lot.datasets import load_dataset, list_supported_datasets, get_dataset_info

def main():
    """
    Main function demonstrating the usage of the dataset interface.
    """
    # List supported datasets
    print("Supported datasets:")
    datasets = list_supported_datasets()
    for dataset in datasets:
        print(f"  - {dataset}")
    print()
    
    # Get dataset information
    print("Dataset information:")
    for dataset in datasets:
        info = get_dataset_info(dataset)
        print(f"  - {dataset}: {info}")
    print()
    
    # Load AQuA dataset
    print("Loading AQuA dataset:")
    data_path = "data/aqua.jsonl"
    if os.path.exists(data_path):
        dataset = load_dataset("aqua", data_path)
        print(f"  - Dataset size: {len(dataset)}")
        
        # Print the first example
        if len(dataset) > 0:
            example = dataset[0]
            print(f"  - First example:")
            print(f"    - Question: {example['question']}")
            print(f"    - Options: {example['options']}")
            print(f"    - Answer: {example['answer']}")
            if 'explanation' in example:
                print(f"    - Explanation: {example['explanation']}")
            
            # Get formatted query
            query = dataset.get_query(0)
            print(f"  - Formatted query: {query}")
    else:
        print(f"  - Dataset file does not exist: {data_path}")

if __name__ == "__main__":
    main() 