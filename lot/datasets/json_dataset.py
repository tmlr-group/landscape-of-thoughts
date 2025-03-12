import json
from typing import Dict, List, Any, Iterator
import os

from .base import BaseDataset

class JsonDataset(BaseDataset):
    """
    A generic dataset class for JSON/JSONL files.
    
    This class provides an interface to use any JSON/JSONL dataset with the Landscape of Thoughts library.
    Users can specify the field names for queries and answers.
    """
    
    def __init__(self, data_path: str, query_field: str = "query", answer_field: str = "answer", is_jsonl: bool = True):
        """
        Initialize the JSON dataset.
        
        Args:
            data_path (str): Path to the dataset file (JSON or JSONL format).
            query_field (str, optional): The field name for queries. Defaults to "query".
            answer_field (str, optional): The field name for answers. Defaults to "answer".
            is_jsonl (bool, optional): Whether the file is in JSONL format. Defaults to True.
        """
        self.data_path = data_path
        self.query_field = query_field
        self.answer_field = answer_field
        self.is_jsonl = is_jsonl
        self.data = []
        
        # Load the dataset
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        if is_jsonl:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    # Ensure the item has the required fields
                    if query_field not in item or answer_field not in item:
                        print(f"Warning: Item missing required fields: {item}")
                        continue
                    self.data.append(item)
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                if isinstance(json_data, list):
                    for item in json_data:
                        # Ensure the item has the required fields
                        if query_field not in item or answer_field not in item:
                            print(f"Warning: Item missing required fields: {item}")
                            continue
                        self.data.append(item)
                else:
                    raise ValueError("JSON file must contain a list of objects")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single example from the dataset.
        
        Args:
            idx (int): The index of the example to retrieve.
            
        Returns:
            Dict[str, Any]: A dictionary containing the example data.
        """
        return self.data[idx]
    
    def __len__(self) -> int:
        """
        Get the number of examples in the dataset.
        
        Returns:
            int: The number of examples.
        """
        return len(self.data)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Get an iterator over the dataset.
        
        Returns:
            Iterator[Dict[str, Any]]: An iterator over the examples in the dataset.
        """
        return iter(self.data)
