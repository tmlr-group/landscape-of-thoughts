import json
import os
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Union

def sample(
    dataset,
    model,
    algorithm,
    num_samples: int = 10,
    start_index: int = 0,
    end_index: Optional[int] = None,
    save_root: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Sample reasoning traces from a dataset using a specified algorithm.
    
    Args:
        dataset: The dataset to sample from.
        model: The model to use for sampling.
        algorithm: The algorithm to use for reasoning.
        num_samples (int): Number of samples to generate per example.
        start_index (int): Index of the first example to sample.
        end_index (Optional[int]): Index of the last example to sample. If None, uses len(dataset).
        save_root (Optional[str]): Root directory to save results. If None, results are not saved.
        
    Returns:
        Tuple[Dict[str, Any], Dict[str, float]]: Features and metrics from sampling.
    """
    # Set end_index if not provided
    if end_index is None:
        end_index = len(dataset)
    
    # Set model for algorithm
    algorithm.set_model(model)
    
    # Initialize features and metrics
    features = {}
    metrics = {"accuracy": []}
    
    # Generate thoughts
    print(f"==> start sampling ...")
    for i in range(start_index, min(end_index, len(dataset))):
        print(f"==> sample: {i}/{min(end_index, len(dataset))-start_index}")
        
        # Get example from dataset
        example = dataset[i]
        
        # Get query from example
        query = example.get('query', example.get('question', ''))
        
        # Get ground truth answer
        gt = example.get('answer', example.get('correct', ''))
        
        # Generate samples
        trial_thoughts = []
        start_time = time.time()
        correct_cnt = 0
        
        for _ in range(num_samples):
            # Do reasoning using the algorithm
            response = algorithm.do_reasoning(query)
            
            # Extract answer (this depends on the dataset and algorithm)
            # For simplicity, we'll assume the answer is the last token that matches A, B, C, D, or E
            matches = re.findall(r'A|B|C|D|E', response)
            pred = matches[-1] if matches else ""
            
            # Process thoughts
            thoughts = [x.strip() for x in response.split("\n") if x.strip()]
            correctness = pred == gt
            if correctness:
                correct_cnt += 1
            
            trial_thoughts.append([thoughts, pred, correctness])
        
        end_time = time.time()
        print(f"==> time consuming: {end_time - start_time:.2f}s")
        print("="*20)
        
        # Save results
        features[i] = {
            "query": query,
            "answer": gt,
            "trial_thoughts": trial_thoughts
        }
        
        # Update metrics
        accuracy = correct_cnt / num_samples
        metrics["accuracy"].append(accuracy)
        
        # Save to file if save_root is provided
        if save_root:
            # Create directory if it doesn't exist
            save_dir = os.path.join(save_root, "thoughts")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Save features
            save_path = os.path.join(save_dir, f"sample_{i}.json")
            with open(save_path, 'w') as f:
                json.dump(features[i], f)
    
    # Calculate average metrics
    metrics["avg_accuracy"] = sum(metrics["accuracy"]) / len(metrics["accuracy"]) if metrics["accuracy"] else 0
    
    return features, metrics 