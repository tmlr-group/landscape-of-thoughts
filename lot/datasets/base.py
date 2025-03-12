from abc import ABC, abstractmethod
from typing import Dict, List, Any, Iterator

class BaseDataset(ABC):
    """
    Base class for all datasets in the Landscape of Thoughts library.
    
    Any dataset used with the library must implement the __getitem__, __len__, and __iter__ methods.
    Users can extend this class to wrap their own datasets.
    """
    
    def __init__(self, dataset_name: str = "unknown"):
        """
        Initialize the base dataset.
        
        Args:
            dataset_name (str, optional): Name of the dataset. Defaults to "unknown".
        """
        self.dataset_name = dataset_name
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single example from the dataset.
        
        Args:
            idx (int): The index of the example to retrieve.
            
        Returns:
            Dict[str, Any]: A dictionary containing the example data.
                            Should include at least 'query' and 'answer' keys.
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Get the number of examples in the dataset.
        
        Returns:
            int: The number of examples.
        """
        pass
    
    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Get an iterator over the dataset.
        
        Returns:
            Iterator[Dict[str, Any]]: An iterator over the examples in the dataset.
        """
        pass
