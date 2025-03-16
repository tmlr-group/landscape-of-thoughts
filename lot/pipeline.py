import json
import os
import time
import pickle as pkl
from typing import Dict, Any, Optional, Tuple

import numpy as np
import plotly.io as pio

from .models import opensource_API_models
from .datasets import load_dataset, extract_answer_from_response, parse_thoughts
from .algorithms import Prompt, MCTS_Task, ToT_Task
from .algorithms.utils import get_tree
from . import get_distance_matrix
from .visualization import draw_landscape, process_landscape_data

def sample_main(
    model_name: str = 'meta-llama/Meta-Llama-3-8B-Instruct-Lite',
    port: int = 8000,
    dataset_name: str = 'aqua',
    data_path: str = 'data/aqua.jsonl',
    method: str = 'cot',
    num_samples: int = 10,
    start_index: int = 0,
    end_index: int = 2,
    prompt_file: Optional[str] = None,
    max_tokens: int = 2048,
    save_root: str = "exp-data"
) -> Tuple[Dict[str, Any], Dict[str, float]]:
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
        max_tokens (int): Maximum number of tokens for model responses.
        save_root (str): Root directory to save results.
        
    Returns:
        Tuple[Dict[str, Any], Dict[str, float]]: Features and metrics from sampling.
    """
    print(f"==> model_name: {model_name}")
    print(f"==> dataset_name: {dataset_name}")
    print(f"==> data_path: {data_path}")
    print(f"==> method: {method}")
    print(f"==> num_samples: {num_samples}")
    print(f"==> start_index: {start_index}")
    print(f"==> end_index: {end_index}")
    print(f"==> save_root: {save_root}")
    
    # Load dataset
    dataset = load_dataset(dataset_name, data_path)
    
    # Initialize model
    model = opensource_API_models(model_name, max_tokens=max_tokens, port=port)
    
    # Initialize algorithm
    if method in ['cot', 'l2m', 'zero-shot-cot']:
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
    features, metrics = sample_with_lot(
        dataset=dataset,
        model=model,
        algorithm=algorithm,
        num_samples=num_samples,
        start_index=start_index,
        end_index=end_index,
        save_root=save_root
    )
    
    # Print metrics
    print(f"Average accuracy: {metrics['avg_accuracy']:.4f}")
    
    return features, metrics

def sample_with_lot(
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
    
    # Get method name
    method = getattr(algorithm, 'method', type(algorithm).__name__)
    
    # Get model name
    model_name = model.model
    model_name_short = model_name.split("/")[-1] if "/" in model_name else model_name
    
    # Generate thoughts
    print(f"==> Start sampling ...")
    for i in range(start_index, min(end_index, len(dataset))):
        print(f"==> Sample: {i}/{min(end_index, len(dataset))-start_index}")
        
        # Check if file already exists
        save_dir = os.path.join(save_root, dataset.dataset_name, "thoughts")
        save_path = os.path.join(save_dir, f"{model_name_short}--{method}--{dataset.dataset_name}--{i}.json")
        if os.path.exists(save_path):
            print(f"==> Skip: {save_path} (already exists)")
            continue
        
        # Get example
        example = dataset[i]
        
        # Get query
        query = dataset.get_query(i)
        
        # Get ground truth answer
        gt = example["answer"]
        
        # Get explanation if available
        explanation = example.get("explanation", "")
        if not explanation and "rationale" in example:
            explanation = example["rationale"]
        
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
        
        # Generate answer options
        answer_full_list = []
        if hasattr(dataset, 'get_answers') and callable(getattr(dataset, 'get_answers')):
            answer_full_list = dataset.get_answers(i)
        elif hasattr(example, 'options') and isinstance(example['options'], list):
            # If options are directly available in the example
            answer_full_list = [f"Answer is: {opt}" for opt in example['options']]
        else:
            # Default format for multiple choice questions
            if gt in ["A", "B", "C", "D", "E"]:
                answer_full_list = [
                    f"Answer is: A",
                    f"Answer is: B",
                    f"Answer is: C",
                    f"Answer is: D",
                    f"Answer is: E"
                ]
        
        # Format the data according to the desired structure
        trial_data = {
            "dataset": dataset.dataset_name,
            "model": model_name,
            "method": method,
            "model_input": query,
            "answers": answer_full_list,
            "answer_gt_full": f"Answer is: {gt}",
            "answer_gt_short": gt,
            "answer_gt_expl": explanation,
            "trial_thoughts": trial_thoughts,
            "accuracy": correct_cnt / num_samples
        }
        
        # Save results
        features[i] = trial_data
        
        # Update metrics
        metrics["accuracy"].append(correct_cnt / num_samples)
        
        # Save to file if save_root is provided
        if save_root:
            # Create directory if it doesn't exist
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Save features
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(trial_data, f, ensure_ascii=False, indent=2)
            
            # Save tree data for search-based methods
            if method in ["ToT_Task", "MCTS_Task"] and 'task' in locals() and 'root' in locals():
                df = get_tree(task, root)
                tree_dir = os.path.join(save_root, dataset.dataset_name, "Tree")
                if not os.path.exists(tree_dir):
                    os.makedirs(tree_dir)
                df_path = os.path.join(tree_dir, f"{model_name_short}--{method}--{dataset.dataset_name}--{i}.json")
                df.to_json(df_path)
    
    # Calculate average metrics
    metrics["avg_accuracy"] = sum(metrics["accuracy"]) / len(metrics["accuracy"]) if metrics["accuracy"] else 0
    
    return features, metrics

def calculate_main(
    model_name: str = 'meta-llama/Meta-Llama-3-8B-Instruct-Lite',
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
    max_tokens: int = 1000,
    save_root: str = "exp-data"
) -> Dict[int, np.ndarray]:
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
        save_root (str): Root directory to save results.
        
    Returns:
        Dict[int, np.ndarray]: Dictionary mapping example indices to their distance matrices.
    """
    print(f"==> model_name: {model_name}")
    print(f"==> dataset_name: {dataset_name}")
    print(f"==> data_path: {data_path}")
    print(f"==> method: {method}")
    print(f"==> start_index: {start_index}")
    print(f"==> end_index: {end_index}")
    print(f"==> save_root: {save_root}")
    
    # Load dataset
    dataset = load_dataset(dataset_name, data_path)
    
    # Initialize model
    model = opensource_API_models(model=model_name, max_tokens=max_tokens, port=port)
    
    # Calculate distance matrices
    distance_matrices = calculate_with_lot(
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

def calculate_with_lot(
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
        
        # Load the trial data from the JSON file
        with open(thoughts_file, 'r', encoding='utf-8') as f:
            trial_data = json.load(f)
        
        # Extract the required fields from the trial data
        model_input = trial_data["model_input"]
        answers = trial_data["answers"]
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

def plot_main(
    model_name: str = 'Meta-Llama-3-8B-Instruct-Lite',
    dataset_name: str = 'aqua',
    method: str = 'cot',
    plot_type: str = 'method',
    save_root: str = "exp-data",
    output_dir: str = "figures/landscape"
) -> bool:
    """
    Main function to plot landscape visualizations of reasoning traces.
    
    Args:
        model_name (str): Name of the model to use.
        dataset_name (str): Name of the dataset to use.
        method (str): The reasoning method used (e.g., 'cot', 'standard').
        plot_type (str): Type of plot ('method' or 'model').
        save_root (str): Root directory where data is stored.
        output_dir (str): Directory to save output figures.
        
    Returns:
        bool: True if plotting was successful.
    """
    print(f"==> model_name: {model_name}")
    print(f"==> dataset_name: {dataset_name}")
    print(f"==> method: {method}")
    print(f"==> plot_type: {plot_type}")
    print(f"==> save_root: {save_root}")
    print(f"==> output_dir: {output_dir}")
    
    # Create methods list
    methods = [method] if method else ['cot', 'l2m', 'mcts', 'tot']
    
    # Process data for landscape visualization
    list_all_T_2D, A_matrix_2D, list_plot_data, list_num_all_thoughts_w_start_list = process_landscape_data(
        model=model_name,
        dataset=dataset_name,
        methods=methods,
        plot_type=plot_type,
        ROOT=save_root
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save plots
    method_idx = 0
    for plot_datas, splited_T_2D, num_all_thoughts_w_start_list in zip(list_plot_data, list_all_T_2D, list_num_all_thoughts_w_start_list):
        # Create the figure
        fig = draw_landscape(
            dataset_name=dataset_name,
            plot_datas=plot_datas,
            splited_T_2D=splited_T_2D,
            A_matrix_2D=A_matrix_2D,
            num_all_thoughts_w_start_list=num_all_thoughts_w_start_list
        )
        
        # Define save path
        save_path = os.path.join(output_dir, f"{model_name}-{dataset_name}-{methods[method_idx]}.png")
        
        # Increment method index if not specific method
        if not method:
            method_idx += 1
        
        # Save the figure
        print(f"==> Saving figure to: {save_path}")
        pio.write_image(fig, save_path, scale=6, width=1500, height=350)
    
    print("==> Plotting complete!")
    return True 