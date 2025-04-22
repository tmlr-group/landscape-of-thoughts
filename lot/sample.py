import json
import os
import time
from typing import Dict, Any, Optional, Tuple


from .models import opensource_API_models
from .datasets import load_dataset, extract_answer_from_response, parse_thoughts
from .algorithms import Prompt, MCTS_Task, ToT_Task
from .algorithms.utils import get_tree
from lot.datasets.dataset_loader import DATASET_TYPES


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

def sample(
    model_name: str = 'meta-llama/Meta-Llama-3-8B-Instruct-Lite',
    port: int = 8000,
    dataset_name: str = 'aqua',
    data_path: str = 'lot/data/aqua.jsonl',
    method: str = 'cot',
    num_samples: int = 10,
    start_index: int = 0,
    end_index: int = 2,
    prompt_file: Optional[str] = None,
    max_tokens: int = 2048,
    save_root: str = "exp-data",
    answer_field: str = "correct",
    options_field: str = "options",
    question_field: str = "question",
    local: bool = False,
    local_api_key: str = "token-abc123"
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
        local (bool): Whether to use local server.
        local_api_key (str): API key for the local server.
        answer_field (str): Field name for the answer.
        options_field (str): Field name for the options.
        question_field (str): Field name for the question.
        
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
    
    # Prepare kwargs for dataset loading - only pass custom fields when not using standard datasets
    dataset_kwargs = {}
    
    # Only add these fields if the dataset is not a predefined one or if a custom field is used
    if dataset_name.lower() not in DATASET_TYPES:
        dataset_kwargs.update({
            'answer_field': answer_field,
            'options_field': options_field,
            'question_field': question_field
        })
    
    # Load dataset
    dataset = load_dataset(
        dataset_name, data_path, **dataset_kwargs
    )
    
    # Check if all files already exist
    model_name_short = model_name.split("/")[-1] if "/" in model_name else model_name
    save_dir = os.path.join(save_root, dataset_name, "thoughts")
    all_files_exist = True
    
    for i in range(start_index, min(end_index, len(dataset))):
        save_path = os.path.join(save_dir, f"{model_name_short}--{method}--{dataset_name}--{i}.json")
        if not os.path.exists(save_path):
            all_files_exist = False
            break
    
    if all_files_exist:
        print(f"==> Skip: All files already exist")
        return None, None

    # Initialize model
    model = opensource_API_models(model_name, max_tokens=max_tokens, local=local, port=port, local_api_key=local_api_key)
    
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
