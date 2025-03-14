import json
import os
import pickle as pkl
import time
from typing import Dict, Optional

import numpy as np
from fire import Fire
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lot.models import opensource_API_models
from lot.datasets import load_dataset
from lot import get_distance_matrix

def calculate(
    dataset,
    model,
    method: str,
    start_index: int = 0,
    end_index: Optional[int] = None,
    topk: int = 10,
    default_distance: int = 10,
    debug: bool = False,
    asyn: bool = False,
    save_root: str = "exp-data"
) -> Dict[int, np.ndarray]:
    """
    Calculate distance matrices for reasoning traces from a dataset.
    
    Args:
        dataset: The dataset containing reasoning traces.
        model: The model to use for calculating perplexities.
        method (str): The reasoning method used (e.g., 'cot', 'standard').
        start_index (int): Index of the first example to process.
        end_index (Optional[int]): Index of the last example to process. If None, uses len(dataset).
        topk (int): Number of top thoughts to consider.
        default_distance (int): Default distance value.
        debug (bool): Whether to run in debug mode.
        asyn (bool): Whether to run asynchronously.
        save_root (str): Root directory to save results.
        
    Returns:
        Dict[int, np.ndarray]: Dictionary mapping example indices to their distance matrices.
    """
    # Set end_index if not provided
    if end_index is None:
        end_index = len(dataset)
    
    # Initialize results dictionary
    distance_matrices = {}
    
    # Get model name for file paths
    model_name = model.model.split('/')[-1]
    
    # Process each example
    print(f"==> Start calculating distance matrices...")
    for i in range(start_index, min(end_index, len(dataset))):
        print(f"==> Processing example: {i}/{min(end_index, len(dataset))-start_index}")        
        
        # Load thoughts - include method in the filename
        thoughts_file = os.path.join(save_root, dataset.dataset_name, "thoughts", f"{model_name}--{method}--{dataset.dataset_name}--{i}.json")
        if not os.path.exists(thoughts_file):
            print(f"==> Thoughts file not found: {thoughts_file}")
            continue
        
        trial_data = json.load(open(thoughts_file, 'r'))
        model_input = trial_data["query"]
        answers = trial_data.get("answers", ["A", "B", "C", "D", "E"])  # Default to common multiple choice options if not provided
        trial_thoughts = trial_data["trial_thoughts"]
        
        # Compute distance matrix - include method in the filename
        save_path = os.path.join(save_root, dataset.dataset_name, "distance_matrix", f"{model_name}--{method}--{dataset.dataset_name}--{i}.pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if os.path.exists(save_path) and not debug:
            print(f"==> Distance matrix exists: {save_path}")
            with open(save_path, 'rb') as f:
                distance_matrices[i] = pkl.load(f)
        else:
            start_time = time.time()
            distance_matrix = get_distance_matrix(
                model, model_input, answers, trial_thoughts, 
                topk=topk, default_distance=default_distance, debug=debug, asyn=asyn
            )
            end_time = time.time()
            print(f"==> Time consumed: {end_time - start_time:.2f}s")
            
            distance_matrices[i] = distance_matrix
            
            if not debug:
                print(f"==> Save distance matrix to: {save_path}")
                with open(save_path, 'wb') as f:
                    pkl.dump(distance_matrix, f)
            
            print(f"==> Distance matrix shape: {distance_matrix.shape}")
            print(f"==> Sample values: {np.round(distance_matrix[:5, :5], 1)}")
        
        print("="*20)
    
    return distance_matrices

def main(
    model_name: str = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
    port: int = 8000,
    dataset_name: str = 'aqua',
    data_path: str = 'data/aqua.jsonl',
    method: str = 'cot',
    start_index: int = 0,
    end_index: int = 2,
    topk: int = 10,
    default_distance: int = 10,
    debug: bool = False,
    asyn: bool = False,
    max_tokens: int = 1000
):
    """
    Main function to calculate distance matrices for reasoning traces.
    
    Args:
        model_name (str): Name of the model to use.
        port (int): Port for the API server.
        dataset_name (str): Name of the dataset to use.
        data_path (str): Path to the dataset file.
        method (str): The reasoning method used (e.g., 'cot', 'standard').
        start_index (int): Index of the first example to process.
        end_index (int): Index of the last example to process.
        topk (int): Number of top thoughts to consider.
        default_distance (int): Default distance value.
        debug (bool): Whether to run in debug mode.
        asyn (bool): Whether to run asynchronously.
        max_tokens (int): Maximum number of tokens for model responses.
    """
    # Load dataset
    dataset = load_dataset(dataset_name, data_path)
    
    # Initialize model
    model = opensource_API_models(model=model_name, max_tokens=max_tokens, port=port)
    
    # Calculate distance matrices
    save_root = "exp-data"
    distance_matrices = calculate(
        dataset=dataset,
        model=model,
        method=method,
        start_index=start_index,
        end_index=end_index,
        topk=topk,
        default_distance=default_distance,
        debug=debug,
        asyn=asyn,
        save_root=save_root
    )
    
    print(f"==> Processed {len(distance_matrices)} examples")
    
    return distance_matrices

if __name__ == "__main__":
    Fire(main) 