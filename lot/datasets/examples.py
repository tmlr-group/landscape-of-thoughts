"""
Examples of how to use the unified dataset interface.

This file contains examples of how to use the unified dataset interface to load
and work with different datasets.
"""

from lot.datasets import get_dataset, list_available_datasets, register_dataset

def example_load_aqua():
    """Example of loading the AQuA dataset."""
    # Path to the AQuA dataset file
    data_path = "path/to/aqua.json"
    
    # Load the dataset using the unified interface
    dataset = get_dataset("aqua", data_path)
    
    # Print the first example
    if len(dataset) > 0:
        print("First example:", dataset[0])
    
    # Print the number of examples
    print(f"Dataset contains {len(dataset)} examples")


def example_load_custom_json():
    """Example of loading a custom JSON dataset."""
    # Path to a custom JSON dataset file
    data_path = "path/to/custom_dataset.json"
    
    # Load the dataset using the unified interface with custom field names
    dataset = get_dataset("json", data_path, 
                         query_field="question", 
                         answer_field="correct_answer",
                         is_jsonl=False)
    
    # Print the first example
    if len(dataset) > 0:
        print("First example:", dataset[0])
    
    # Print the number of examples
    print(f"Dataset contains {len(dataset)} examples")

def example_register_custom_dataset():
    """Example of registering a custom dataset class."""
    from lot.datasets import BaseDataset
    
    # Define a custom dataset class
    class MyCustomDataset(BaseDataset):
        def __init__(self, data_path, **kwargs):
            self.data = [{"query": "Custom question 1", "answer": "Custom answer 1"}]
        
        def __getitem__(self, idx):
            return self.data[idx]
        
        def __len__(self):
            return len(self.data)
        
        def __iter__(self):
            return iter(self.data)
    
    # Register the custom dataset
    register_dataset("my_custom", MyCustomDataset)
    
    # Now we can load it using the unified interface
    dataset = get_dataset("my_custom", "dummy_path")
    
    # Print the first example
    print("Custom dataset example:", dataset[0])

def example_list_datasets():
    """Example of listing all available datasets."""
    available_datasets = list_available_datasets()
    print("Available datasets:")
    for name, dataset_class in available_datasets.items():
        print(f"  - {name}: {dataset_class.__name__}")

if __name__ == "__main__":
    # Uncomment the examples you want to run
    # example_load_commonsenseqa()
    # example_load_strategyqa()
    # example_load_custom_json()
    # example_register_custom_dataset()
    example_list_datasets() 