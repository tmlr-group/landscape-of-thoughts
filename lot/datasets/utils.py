"""
Dataset processing utility functions.
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Union, Callable

def load_json_file(file_path: str, is_jsonl: bool = False) -> List[Dict[str, Any]]:
    """
    Load a JSON or JSONL file.
    
    Args:
        file_path (str): File path.
        is_jsonl (bool): Whether it's a JSONL format.
        
    Returns:
        List[Dict[str, Any]]: Loaded data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the JSON format is incorrect.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
    
    data = []
    
    if is_jsonl:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            if isinstance(json_data, list):
                data = json_data
            else:
                data = [json_data]
    
    return data

def get_nested_value(obj: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a value from a nested dictionary.
    
    Args:
        obj (Dict[str, Any]): Dictionary object.
        key_path (str): Key path, such as 'a.b.c'.
        default (Any): Default value to return if the key does not exist.
        
    Returns:
        Any: Found value, or default value.
    """
    if not key_path:
        return default
    
    keys = key_path.split('.')
    current = obj
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current

def format_options(options: Any, formatter: Optional[Callable] = None) -> List[str]:
    """
    Format options list.
    
    Args:
        options (Any): Options data.
        formatter (Optional[Callable]): Formatting function.
        
    Returns:
        List[str]: Formatted options list.
    """
    if formatter is not None:
        return formatter(options)
    
    if isinstance(options, list):
        return options
    
    return []

def format_answer(answer: Any, formatter: Optional[Callable] = None) -> str:
    """
    Format answer.
    
    Args:
        answer (Any): Answer data.
        formatter (Optional[Callable]): Formatting function.
        
    Returns:
        str: Formatted answer.
    """
    if formatter is not None:
        return formatter(answer)
    
    return str(answer)

def extract_answer_from_response(response: str, pattern: str = r'A|B|C|D|E') -> str:
    """
    Extract answer from response.
    
    Args:
        response (str): Model response text.
        pattern (str): Regular expression pattern.
        
    Returns:
        str: Extracted answer, or empty string if not found.
    """
    matches = re.findall(pattern, response)
    return matches[-1] if matches else ""

def parse_thoughts(response: str) -> List[str]:
    """
    Parse thinking process.
    
    Args:
        response (str): Model response text.
        
    Returns:
        List[str]: Parsed thinking steps list.
    """
    return [line.strip() for line in response.split('\n') if line.strip()]

def save_results(
    results: Dict[str, Any],
    save_path: str,
    create_dir: bool = True,
    dataset_name: str = None,
    model_name: str = None,
    method: str = None
) -> None:
    """
    Save results to file.
    
    Args:
        results (Dict[str, Any]): Results data.
        save_path (str): Save path.
        create_dir (bool): Whether to create directory.
        dataset_name (str, optional): Name of the dataset.
        model_name (str, optional): Name of the model.
        method (str, optional): Method used for reasoning.
    """
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Extract information from the results
    query = results.get("query", "")
    answer_gt = results.get("answer", "")
    trial_thoughts = results.get("trial_thoughts", [])
    
    # Format the data according to the desired structure
    formatted_results = {
        "dataset": dataset_name,
        "model": model_name,
        "method": method,
        "model_input": query,
        "answers": [],  # Will be populated based on the dataset
        "answer_gt_full": f"Answer is: {answer_gt}",
        "answer_gt_short": answer_gt,
        "answer_gt_expl": results.get("explanation", ""),
        "trial_thoughts": trial_thoughts
    }
    
    # Generate answer options if not provided
    if "answers" not in results:
        # Default format for multiple choice questions
        if answer_gt in ["A", "B", "C", "D", "E"]:
            formatted_results["answers"] = [
                f"Answer is: A",
                f"Answer is: B",
                f"Answer is: C",
                f"Answer is: D",
                f"Answer is: E"
            ]
    else:
        formatted_results["answers"] = results["answers"]
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_results, f, ensure_ascii=False, indent=2) 