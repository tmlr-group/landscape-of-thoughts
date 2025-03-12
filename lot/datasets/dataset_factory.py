from typing import Dict, Any, Optional, Union, Type
import os

from .base import BaseDataset
from .aqua import AQuA
from .commonsenseqa import CommonsenseQA
from .mmlu import MMLU
from .strategyqa import StrategyQA
from .json_dataset import JsonDataset

# Registry of dataset classes
DATASET_REGISTRY = {
    'aqua': AQuA,
    'commonsenseqa': CommonsenseQA,
    'mmlu': MMLU,
    'strategyqa': StrategyQA,
    'json': JsonDataset,
}

def get_dataset(dataset_name: str, data_path: str, **kwargs) -> BaseDataset:
    """
    Factory function to get a dataset by name.
    
    Args:
        dataset_name (str): Name of the dataset to load. Must be one of the registered datasets.
        data_path (str): Path to the dataset file.
        **kwargs: Additional arguments to pass to the dataset constructor.
        
    Returns:
        BaseDataset: An instance of the requested dataset.
        
    Raises:
        ValueError: If the dataset name is not recognized.
        FileNotFoundError: If the dataset file does not exist.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(DATASET_REGISTRY.keys())}")
    
    dataset_class = DATASET_REGISTRY[dataset_name]
    return dataset_class(data_path, **kwargs)

def list_available_datasets() -> Dict[str, Type[BaseDataset]]:
    """
    Get a dictionary of all available datasets.
    
    Returns:
        Dict[str, Type[BaseDataset]]: A dictionary mapping dataset names to their classes.
    """
    return DATASET_REGISTRY.copy()

def register_dataset(name: str, dataset_class: Type[BaseDataset]) -> None:
    """
    Register a new dataset class.
    
    Args:
        name (str): Name to register the dataset under.
        dataset_class (Type[BaseDataset]): The dataset class to register.
        
    Raises:
        ValueError: If a dataset with the same name is already registered.
    """
    if name in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' is already registered")
    
    DATASET_REGISTRY[name] = dataset_class 