#!/usr/bin/env python
"""
Example of how to wrap a custom model for use with the lot module.

This script demonstrates how to create a custom model wrapper that can be used
with the lot module for sampling reasoning traces.
"""

import os
import json
from fire import Fire

import lot
from lot.models.base import BaseModel
from lot.algorithms import CoT
from lot.datasets import AQuA


class CustomModel(BaseModel):
    """
    Example of a custom model wrapper for the lot module.
    
    This class demonstrates how to wrap a custom model for use with the lot module.
    Users need to implement the __call__ and get_likelihood methods.
    """
    
    def __init__(self, model_name="your-custom-model", **kwargs):
        """
        Initialize the custom model.
        
        Args:
            model_name: Name of the custom model.
            **kwargs: Additional arguments for the model.
        """
        super().__init__(model_name=model_name, **kwargs)
        
        # Initialize your custom model here
        # For example:
        # self.model = YourCustomModel(...)
        print(f"Initialized custom model: {model_name}")
        
    def __call__(self, prompt):
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The prompt to generate a response for.
            
        Returns:
            str: The generated response.
        """
        # Implement your custom model's generation logic here
        # For example:
        # response = self.model.generate(prompt)
        
        # For demonstration purposes, we'll just return a mock response
        print(f"Generating response for prompt: {prompt[:50]}...")
        return "This is a mock response from the custom model."
    
    def get_likelihood(self, prompt, continuation):
        """
        Calculate the likelihood of the continuation given the prompt.
        
        Args:
            prompt: The prompt.
            continuation: The continuation to calculate the likelihood for.
            
        Returns:
            float: The likelihood of the continuation.
        """
        # Implement your custom model's likelihood calculation logic here
        # For example:
        # likelihood = self.model.calculate_likelihood(prompt, continuation)
        
        # For demonstration purposes, we'll just return a mock likelihood
        print(f"Calculating likelihood for continuation: {continuation[:50]}...")
        return 0.8  # Mock likelihood


class CustomDataset(lot.datasets.base.BaseDataset):
    """
    Example of a custom dataset wrapper for the lot module.
    
    This class demonstrates how to wrap a custom dataset for use with the lot module.
    Users need to implement the __getitem__ and __len__ methods.
    """
    
    def __init__(self, data_path, **kwargs):
        """
        Initialize the custom dataset.
        
        Args:
            data_path: Path to the dataset.
            **kwargs: Additional arguments for the dataset.
        """
        super().__init__(**kwargs)
        self.data_path = data_path
        
        # Load your custom dataset here
        # For example:
        # self.data = load_custom_dataset(data_path)
        
        # For demonstration purposes, we'll just create a mock dataset
        self.data = [
            {"query": "What is 2+2?", "answer": "4"},
            {"query": "What is the capital of France?", "answer": "Paris"},
            {"query": "What is the square root of 16?", "answer": "4"}
        ]
        
        print(f"Loaded custom dataset from: {data_path}")
        
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item to get.
            
        Returns:
            dict: The item at the given index.
        """
        if isinstance(idx, slice):
            # Handle slicing
            return [self.data[i] for i in range(*idx.indices(len(self)))]
        else:
            # Handle single index
            return self.data[idx]
    
    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: The length of the dataset.
        """
        return len(self.data)


def main(
    dataset_path: str = 'data/custom_dataset.json',
    method: str = 'cot',
    samples: int = 5
):
    """
    Demonstrate how to use a custom model and dataset with the lot module.
    
    Args:
        dataset_path: Path to the custom dataset.
        method: Method to use for reasoning (cot, tot).
        samples: Number of samples per query.
    """
    print(f"==> dataset_path: {dataset_path}\n==> method: {method}\n==> samples_cnt: {samples}\n")

    # 1. Initialize custom model
    model = CustomModel(model_name="my-custom-model")

    # 2. Initialize custom dataset
    dataset = CustomDataset(data_path=dataset_path)

    # 3. Initialize algorithm
    algorithm = CoT(prompt_file="prompts/custom_cot.txt")

    # 4. Sample reasoning traces
    print(f"==> start sampling ...")
    features, metrics = lot.sample(dataset, model, algorithm, num_sample=samples)

    # 5. Save the results
    save_root = "exp-data-custom"
    
    # Create directory if it doesn't exist
    os.makedirs(f"{save_root}/thoughts", exist_ok=True)
    
    # Save the features and metrics
    for i, (feature, accuracy) in enumerate(zip(features, metrics['accuracy'])):
        save_path = f"{save_root}/thoughts/custom-model--{method}--custom-dataset--{i}.json"
        
        # Convert features to a serializable format
        serializable_features = []
        for step_feature in feature:
            serializable_features.append(step_feature.tolist())
        
        # Create the trial data
        trial_data = {
            "dataset": "custom",
            "model": "custom-model",
            "method": method,
            "features": serializable_features,
            "accuracy": float(accuracy),
            "consistency": float(metrics['consistency'][i]),
            "uncertainty": float(metrics['uncertainty'][i]),
            "perplexity": float(metrics['perplexity'][i])
        }
        
        # Save the trial data
        with open(save_path, 'w') as f:
            json.dump(trial_data, f)
    
    print(f"==> Sampling complete. Results saved to {save_root}/thoughts/")
    
    # 6. Visualize the results
    lot.visualize(features, metrics=metrics, file_pattern=f"{save_root}/visualization_%d.png")
    print(f"==> Visualization saved to {save_root}/")


if __name__ == "__main__":
    Fire(main) 