import json
from typing import Dict, List, Any, Iterator, Optional, Callable, Union
import os

from .base import BaseDataset
from .utils import get_nested_value, format_options, format_answer, load_json_file

class JsonDataset(BaseDataset):
    """
    Generic JSON/JSONL dataset class.
    
    This class provides an interface for using any JSON/JSONL dataset with the Landscape of Thoughts library.
    Users can specify field names for queries, options, and answers.
    """
    
    def __init__(
        self,
        data_path: str,
        question_field: str = "question",
        options_field: Optional[str] = "options",
        answer_field: str = "answer",
        explanation_field: Optional[str] = None,
        is_jsonl: bool = True,
        options_format: Optional[Callable] = None,
        dataset_name: Optional[str] = None
    ):
        """
        Initialize the JSON dataset.
        
        Args:
            data_path (str): Dataset file path (JSON or JSONL format).
            question_field (str, optional): Question field name. Defaults to "question".
            options_field (Optional[str], optional): Options field name. Defaults to "options".
            answer_field (str, optional): Answer field name. Defaults to "answer".
            explanation_field (Optional[str], optional): Explanation field name. Defaults to None.
            is_jsonl (bool, optional): Whether it's a JSONL format. Defaults to False.
            options_format (Optional[Callable], optional): Options formatting function. Defaults to None.
            answer_format (Optional[Callable], optional): Answer formatting function. Defaults to None.
            dataset_name (Optional[str], optional): Name of the dataset. If None, derived from data_path. Defaults to None.
        """
        # Set dataset name
        if dataset_name is not None:
            self.dataset_name = dataset_name
        else:
            # Extract dataset name from file path
            base_name = os.path.basename(data_path)
            self.dataset_name = os.path.splitext(base_name)[0]
            
        # Call parent class's __init__ method
        super().__init__(dataset_name=self.dataset_name)
        
        self.data_path = data_path
        self.question_field = question_field
        self.options_field = options_field
        self.answer_field = answer_field
        self.explanation_field = explanation_field
        self.is_jsonl = is_jsonl
        self.options_format = options_format
        
        # Prompt information (will be set by dataset_loader)
        self.prompts = {}
        self.answer_pattern = r'A|B|C|D|E'
        
        # Load dataset
        self.data = load_json_file(data_path, is_jsonl)
        
        # Process data
        self.processed_data = []
        for item in self.data:
            processed_item = self._process_item(item)
            if processed_item:
                self.processed_data.append(processed_item)
    
    def _process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single data item.
        
        Args:
            item (Dict[str, Any]): Original data item.
            
        Returns:
            Optional[Dict[str, Any]]: Processed data item, or None if missing required fields.
        """
        # Get question
        question = get_nested_value(item, self.question_field)
        if question is None:
            print(f"Warning: Item missing question field: {item}")
            return None
        
        # Get options
        options = []
        if self.options_field:
            raw_options = get_nested_value(item, self.options_field)
            options = format_options(raw_options, self.options_format)
        
        # Get answer
        answer = get_nested_value(item, self.answer_field)
        if answer is None:
            print(f"Warning: Item missing answer field: {item}")
            return None
        
        
        # Get explanation (if any)
        explanation = None
        if self.explanation_field:
            explanation = get_nested_value(item, self.explanation_field)
        
        # Create processed item
        processed_item = {
            "question": question,
            "options": options,
            "answer": answer,
        }
        
        if explanation:
            processed_item["explanation"] = explanation
        
        # Keep original data
        processed_item["_original"] = item
        
        return processed_item
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single example from the dataset.
        
        Args:
            idx (int): The index of the example to retrieve.
            
        Returns:
            Dict[str, Any]: A dictionary containing the example data.
        """
        return self.processed_data[idx]
    
    def __len__(self) -> int:
        """
        Get the number of examples in the dataset.
        
        Returns:
            int: The number of examples.
        """
        return len(self.processed_data)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Get an iterator over the dataset.
        
        Returns:
            Iterator[Dict[str, Any]]: An iterator over the examples in the dataset.
        """
        return iter(self.processed_data)
    
    def get_original_data(self, idx: int) -> Dict[str, Any]:
        """
        Get the original data item.
        
        Args:
            idx (int): The index of the item to retrieve.
            
        Returns:
            Dict[str, Any]: The original data item.
        """
        return self.data[idx]
    
    def get_query(self, idx: int) -> str:
        """
        Get a formatted query string.
        
        Args:
            idx (int): The index of the example to retrieve.
            
        Returns:
            str: The formatted query string.
        """
        item = self.processed_data[idx]
        question = item["question"]
        options = item.get("options", [])
        
        if options:
            return f"{question}\nOptions: {' '.join(options)}"
        else:
            return question
    
    def get_prompt(self, method: str, default: str = None) -> str:
        """
        Get the prompt template for a specific reasoning method.
        
        Args:
            method (str): Reasoning method (cot, standard, tot, mcts).
            default (str, optional): Default prompt template if not found. Defaults to None.
            
        Returns:
            str: The prompt template, or default if not found.
        """
        method = method.lower()
        
        if method in self.prompts:
            return self.prompts[method]
        
        return default
    
    def get_answer(self, idx: int) -> str:
        """
        Get the answer for the dataset.
        
        Returns:
            str: The answer.
        """
        return self.processed_data[idx]["answer"]

    def format_prompt(self, idx: int, method: str, default_prompt: str = None) -> str:
        """
        Format a prompt for a specific example and reasoning method.
        
        Args:
            idx (int): The index of the example to retrieve.
            method (str): Reasoning method (cot, standard, tot, mcts).
            default_prompt (str, optional): Default prompt template if not found. Defaults to None.
            
        Returns:
            str: The formatted prompt.
        """
        query = self.get_query(idx)
        prompt_template = self.get_prompt(method, default_prompt)
        
        if prompt_template:
            return prompt_template.format(query=query)
        
        # If no prompt template is found, return the query as is
        return query
