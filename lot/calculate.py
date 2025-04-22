import json
import os
import time
import pickle as pkl
import numpy as np
from typing import Dict, Optional, List, Tuple

from .models import opensource_API_models
from .datasets import load_dataset

from tqdm import tqdm

def get_distance_matrix(
    model,
    model_input: str,
    answers: List[str],
    trial_thoughts: List[Tuple[List[str], str, bool]],
    default_distance: int = 10,
    debug: bool = False,
) -> np.ndarray:
    """
    Calculate the distance matrix for a set of reasoning traces.
    
    Args:
        model: The language model.
        model_input: The input question.
        answers: The list of candidate answers.
        trial_thoughts: The list of chain of thoughts; each chain contains a list of split thoughts.
        default_distance (int): Default distance value.
        debug (bool): Whether to run in debug mode.
        
    Returns:
        np.ndarray: The distance matrix of size (num_all_thoughts+num_anchors, num_anchors)
    """
    # parse thoughts
    num_thoughts_each_chain = [len(thoughts) for [thoughts, _, _] in trial_thoughts]
    all_answers = [answer for [thoughts, answer, _] in trial_thoughts]
    all_thoughts = []
    for [thoughts, _, _] in trial_thoughts:
        all_thoughts += thoughts
    
    all_thoughts = np.array(all_thoughts)
    num_all_thoughts = len(all_thoughts)
    
    # parse anchors
    anchors = [model_input] + answers # question and answers
    num_anchors = len(anchors)
    anchors_idx_y = [i for i in range(num_anchors)]
    anchors_idx_x = [(num_all_thoughts + i) for i in range(num_anchors)]
    
    # initialize the distance matrix
    distance_matrix = np.ones((num_all_thoughts+num_anchors, num_anchors)) * default_distance
    
    # calculate the distance matrix
    for chain_idx in tqdm(range(len(trial_thoughts)), ncols=50, desc="Processing chains"):
        # prepare
        thoughts, _, _ = trial_thoughts[chain_idx]
        start_idx, end_idx = sum(num_thoughts_each_chain[:chain_idx]), sum(num_thoughts_each_chain[:chain_idx+1])
        num_thoughts = num_thoughts_each_chain[chain_idx]
        
        # split the thoughts into states
        # note that the final answer (e.g., "Answer is: A") is included
        states_with_question = []
        states_without_question = []
        for i in range(len(thoughts)): 
            state_with_question = model_input + " " + " ".join(thoughts[:i+1])
            state_without_question = " ".join(thoughts[:i+1])
            states_with_question.append(state_with_question)
            states_without_question.append(state_without_question)
        assert len(states_with_question) == len(states_without_question)
        assert len(states_with_question) == num_thoughts

        # [1] compute p(state|question): X -> S_i
        #######################################
        perplexity_states_question = np.ones(len(states_without_question))*1 if debug else model.get_perplexity(model_input, states_without_question)
        distance_matrix[start_idx:end_idx, anchors_idx_y[0]] = np.array(perplexity_states_question)
        
        # [2] compute p(answer|state): S_i -> Y_1, Y_2, ..., Y_N
        #######################################
        for state_idx, state in enumerate(states_with_question):
            target_thoughts = answers.copy() # there is no next thought
            target_thoughts_idx = anchors_idx_y[1:]
            perplexity = np.ones(len(target_thoughts))*2 if debug else model.get_perplexity(state, target_thoughts)
            distance_matrix[start_idx+state_idx, target_thoughts_idx] = np.array(perplexity)
    
    # [3] get the anchors' coordinates
    # p(answer-1|question), p(answer-2|question), ..., p(answer-C|question) 
    #######################################
    perplexity = np.ones(len(answers))*3 if debug else model.get_perplexity(model_input, answers)
    distance_matrix[anchors_idx_x[0], anchors_idx_y[1:]] = np.array(perplexity) 
    distance_matrix[anchors_idx_x[1:], anchors_idx_y[0]] = np.array(perplexity)
    distance_matrix[anchors_idx_x[0], anchors_idx_y[0]] = 0 
    distance_matrix[anchors_idx_x[1:], anchors_idx_y[1:]] = 0 # make the diagonal be zeros
    
    return distance_matrix 


def calculate_with_lot(
    dataset,
    model,
    method: str,
    start_index: int = 0,
    end_index: Optional[int] = None,
    default_distance: int = 10,
    debug: bool = False,
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
        default_distance (int): Default distance value.
        debug (bool): Whether to run in debug mode.
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
                default_distance=default_distance, debug=debug
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



def calculate(
    model_name: str = 'meta-llama/Meta-Llama-3-8B-Instruct-Lite',
    port: int = 8000,
    dataset_name: str = 'aqua',
    data_path: str = 'data/aqua.jsonl',
    method: str = 'cot',
    start_index: int = 0,
    end_index: int = 2,
    default_distance: int = 10,
    debug: bool = False,
    max_tokens: int = 1000,
    save_root: str = "exp-data",
    answer_field: str = "correct",
    options_field: str = "options",
    question_field: str = "question",
    local: bool = False,
    local_api_key: str = "token-abc123",
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
        default_distance (int): Default distance value.
        debug (bool): Whether to run in debug mode.
        max_tokens (int): Maximum number of tokens for model responses.
        save_root (str): Root directory to save results.
        local (bool): Whether to use local server.
        local_api_key (str): API key for the local server.
        answer_field (str): Field name for the answer.
        options_field (str): Field name for the options.
        question_field (str): Field name for the question.
        
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
    dataset = load_dataset(dataset_name, data_path, answer_field=answer_field, options_field=options_field, question_field=question_field)

    # Check if all files already exist
    model_name_short = model_name.split("/")[-1] if "/" in model_name else model_name
    save_dir = os.path.join(save_root, dataset_name, "distance_matrix")
    all_files_exist = True
    for i in range(start_index, min(end_index, len(dataset))):
        save_path = os.path.join(save_dir, f"{model_name_short}--{method}--{dataset_name}--{i}.pkl")
        if not os.path.exists(save_path):
            all_files_exist = False
            break
    
    if all_files_exist:
        print(f"==> Skip: All files already exist")
        return None, None

    # Initialize model
    model = opensource_API_models(model=model_name, max_tokens=max_tokens, port=port, local=local, local_api_key=local_api_key)
    
    # Calculate distance matrices
    distance_matrices = calculate_with_lot(
        dataset=dataset,
        model=model,
        method=method,
        start_index=start_index,
        end_index=end_index,
        default_distance=default_distance,
        debug=debug,
        save_root=save_root
    )
    
    print(f"==> Processed {len(distance_matrices)} examples")
    
    return distance_matrices
