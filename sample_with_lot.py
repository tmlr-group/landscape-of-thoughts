import json
import os
import re
import time
from typing import Dict, List, Any, Optional, Tuple

from fire import Fire
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lot.models import opensource_API_models
from lot.datasets import load_dataset, extract_answer_from_response, parse_thoughts, save_results
from lot.algorithms import Prompt, MCTS_Task, ToT_Task

def sample(
    dataset,
    model,
    algorithm,
    num_samples: int = 10,
    start_index: int = 0,
    end_index: Optional[int] = None,
    save_root: str = "exp-data"
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
        save_root (str): Root directory to save results.
        
    Returns:
        Tuple[Dict[str, Any], Dict[str, float]]: Features and metrics from sampling.
    """
    # Set end_index if not provided
    if end_index is None:
        end_index = len(dataset)
    
    # Set model for algorithm if it's a Prompt algorithm
    if hasattr(algorithm, 'set_model'):
        algorithm.set_model(model)
    
    # Initialize features and metrics
    features = {}
    metrics = {"accuracy": []}
    
    # Generate thoughts
    print(f"==> Start sampling ...")
    for i in range(start_index, min(end_index, len(dataset))):
        print(f"==> Sample: {i}/{min(end_index, len(dataset))-start_index}")
        
        # Get example
        example = dataset[i]
        
        # Get query
        query = dataset.get_query(i)
        
        # Get ground truth answer
        gt = example["answer"]
        
        # Generate samples
        trial_thoughts = []
        start_time = time.time()
        correct_cnt = 0
        
        for _ in range(num_samples):
            # Do reasoning using the algorithm
            if isinstance(algorithm, Prompt):
                # For Prompt algorithm, use do_reasoning method
                response = algorithm.do_reasoning(query)
            elif algorithm == MCTS_Task:
                # For MCTS algorithm, create a new task instance and run it
                task = MCTS_Task(query, model=model, propose_method='llama', value_method='llama', lang='en', iteration_limit=2)
                output, root = task.run()
                response = "\n".join(output[field] for field in ['content', 'solution', 'summary'] if field in output)
            elif algorithm == ToT_Task:
                # For ToT algorithm, create a new task instance and run it
                task = ToT_Task(query, model=model, propose_method='llama', value_method='llama', algorithm='dfs', lang='en', max_depth=5)
                output, root = task.run()
                response = "\n".join(output[field] for field in ['content', 'solution', 'summary'] if field in output)
            else:
                raise ValueError(f"Unsupported algorithm type: {type(algorithm)}")
            
            # Extract answer using dataset-specific pattern
            pred = extract_answer_from_response(response, pattern=dataset.answer_pattern)
            
            # Process thoughts
            thoughts = parse_thoughts(response)
            correctness = pred == gt
            if correctness:
                correct_cnt += 1
            
            trial_thoughts.append([thoughts, pred, correctness])
        
        end_time = time.time()
        print(f"==> Time consumed: {end_time - start_time:.2f}s")
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
            model_name = model.model.split("/")[-1] if "/" in model.model else model.model
            method = getattr(algorithm, 'method', type(algorithm).__name__)
            save_path = os.path.join(save_dir, f"{model_name}--{method}--{dataset.dataset_name}--{i}.json")
            save_results(features[i], save_path)
    
    # Calculate average metrics
    metrics["avg_accuracy"] = sum(metrics["accuracy"]) / len(metrics["accuracy"]) if metrics["accuracy"] else 0
    
    return features, metrics

def main(
    model_name: str = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
    port: int = 8000,
    dataset_name: str = 'aqua',
    data_path: str = 'data/aqua.jsonl',
    method: str = 'cot',
    num_samples: int = 10,
    start_index: int = 0,
    end_index: int = 2,
    prompt_file: Optional[str] = None,
    max_tokens: int = 2048
):
    """
    Main function to run the sampling process.
    
    Args:
        model_name (str): Name of the model to use.
        port (int): Port for the API server.
        dataset_name (str): Name of the dataset to use.
        data_path (str): Path to the dataset file.
        method (str): Method to use for reasoning (cot, standard, tot, mcts).
        num_samples (int): Number of samples to generate per example.
        start_index (int): Index of the first example to sample.
        end_index (int): Index of the last example to sample.
        prompt_file (Optional[str]): Path to a prompt file for the algorithm.
    """
    # Load dataset
    dataset = load_dataset(dataset_name, data_path)
    
    # Initialize model
    model = opensource_API_models(model_name, max_tokens=max_tokens, port=port)
    
    # Initialize algorithm
    if method in ['cot', 'l2m', 'zero-shot-cot', 'standard']:
        algorithm = Prompt(method=method)
        # Set examples to empty list by default
        algorithm.set_example([])
        
        if prompt_file:
            with open(prompt_file, 'r') as f:
                prompt_template = f.read()
        else:
            # Use dataset-specific prompt if available
            prompt_template = dataset.get_prompt(method, "Please answer the following question by reasoning step-by-step.\n\n{question}\n\nLet's think through this step-by-step:")
        
        algorithm.set_question_template(prompt_template)
    elif method == 'tot':
        algorithm = ToT_Task
    elif method == 'mcts':
        algorithm = MCTS_Task
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Sample from dataset
    save_root = "exp-data"
    features, metrics = sample(
        dataset=dataset,
        model=model,
        algorithm=algorithm,
        num_samples=num_samples,
        start_index=start_index,
        end_index=end_index,
        save_root=f"{save_root}/{dataset_name}"
    )
    
    # Print metrics
    print(f"Average accuracy: {metrics['avg_accuracy']:.4f}")
    
    return features, metrics

if __name__ == "__main__":
    Fire(main)