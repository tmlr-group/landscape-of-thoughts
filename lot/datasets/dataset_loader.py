"""
Unified dataset loading interface, supporting various types of datasets.
"""

import json
import os
from typing import Dict, Any, Optional, List, Union, Tuple

from .base import BaseDataset
from .json_dataset import JsonDataset
from .prompt import get_prompt, get_answer_pattern, DATASET_PROMPTS, DATASET_PATTERNS, ANSWER_IDX_MAPPER

# Dataset type mapping
DATASET_TYPES = {
    'mmlu': 'json',
    'aqua': 'json',
    'commonsenseqa': 'json',
    'strategyqa': 'json',
}

# Dataset field mapping
DATASET_FIELDS = {
    'mmlu': {
        'question_field': 'question',
        'options_field': 'options',
        'answer_field': 'answer',
    },
    'aqua': {
        'question_field': 'question',
        'options_field': 'options',
        'answer_field': 'correct',
        'explanation_field': 'rationale',
    },
    'commonsenseqa': {
        'question_field': 'question.stem',
        'options_field': 'question.choices',
        'answer_field': 'answerKey',
        'options_format': lambda choices: [f"{item['label']}. {item['text']}" for item in choices],
    },
    'strategyqa': {
        'question_field': 'question',
        'options_field': None,  # Special handling
        'answer_field': 'answer',
        'options_format': lambda _: ["A. yes", "B. no"],
        'answer_format': lambda answer: "A" if answer else "B",
    },
}

def load_dataset(
    dataset_name: str,
    data_path: str,
    **kwargs
) -> BaseDataset:
    """
    Unified dataset loading function.
    
    Args:
        dataset_name (str): Dataset name, such as 'mmlu', 'aqua', etc.
        data_path (str): Dataset file path.
        **kwargs: Other parameters to be passed to the specific dataset loading function.
        
    Returns:
        BaseDataset: The loaded dataset object.
        
    Raises:
        ValueError: If the dataset name is not supported or the file does not exist.
    """
    if not os.path.exists(data_path):
        raise ValueError(f"Dataset file does not exist: {data_path}")
    
    dataset_name = dataset_name.lower()
    
    # Check if it's a known dataset type
    if dataset_name not in DATASET_TYPES:
        # If not a known type, try to determine by file extension
        _, ext = os.path.splitext(data_path)
        if ext in ['.json', '.jsonl']:
            return JsonDataset(data_path, **kwargs)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_name}, file extension: {ext}")
    
    # Load based on dataset type
    dataset_type = DATASET_TYPES[dataset_name]
    
    if dataset_type == 'json':
        # Load JSON format dataset
        dataset = load_json_dataset(dataset_name, data_path, **kwargs)
        
        # Attach prompt information to the dataset
        dataset.prompts = get_dataset_prompts(dataset_name)
        dataset.answer_pattern = get_answer_pattern(dataset_name)
        
        return dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def load_json_dataset(
    dataset_name: str,
    data_path: str,
    **kwargs
) -> JsonDataset:
    """
    Load a JSON format dataset.
    
    Args:
        dataset_name (str): Dataset name.
        data_path (str): Dataset file path.
        **kwargs: Other parameters to be passed to the JsonDataset constructor.
        
    Returns:
        JsonDataset: The loaded JSON dataset object.
    """
    # Get dataset field mapping
    fields = DATASET_FIELDS.get(dataset_name, {})
    for key, value in kwargs.items():
        if key in fields:
            fields[key] = value
        else:
            raise ValueError(f"Invalid field: {key} for dataset {dataset_name}")
    
    # Pop the keys from kwargs if they were used in the fields mapping above
    for key in list(fields.keys()):
        if key in kwargs:
            kwargs.pop(key)

    # Determine if it's a JSONL format
    is_jsonl = data_path.endswith('.jsonl')
    
    # Create JsonDataset object
    dataset = JsonDataset(
        data_path=data_path,
        is_jsonl=is_jsonl,
        dataset_name=dataset_name,
        **fields,
        **kwargs
    )
    
    return dataset

def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get dataset metadata.
    
    Args:
        dataset_name (str): Dataset name.
        
    Returns:
        Dict[str, Any]: Dataset metadata.
    """
    if dataset_name not in DATASET_TYPES:
        return {}
    
    return {
        'type': DATASET_TYPES[dataset_name],
        'fields': DATASET_FIELDS.get(dataset_name, {}),
        'prompts': get_dataset_prompts(dataset_name),
        'answer_pattern': get_answer_pattern(dataset_name),
    }

def get_dataset_prompts(dataset_name: str) -> Dict[str, str]:
    """
    Get all prompts for a specific dataset.
    
    Args:
        dataset_name (str): Dataset name.
        
    Returns:
        Dict[str, str]: Dictionary of prompts for different reasoning methods.
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name in DATASET_PROMPTS:
        return DATASET_PROMPTS[dataset_name]
    
    return {}

def list_supported_datasets() -> List[str]:
    """
    List all supported datasets.
    
    Returns:
        List[str]: List of supported dataset names.
    """
    return list(DATASET_TYPES.keys()) 