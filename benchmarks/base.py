from abc import ABC, abstractmethod

class Dataset(ABC):
    @abstractmethod
    def get_dataset(self, 
                    **kwargs) -> str:
        """
        Retrieves the dataset specified by name

        Returns:
        str: The specified dataset.
        """
         
    @abstractmethod
    def get_example(self, 
                    shots: int, 
                    algorithm: str, 
                    **kwargs) -> list[str]:
        """
        Retrieves the example specified by name

        :param shots:
        :param algorithm:
        """
        
    @abstractmethod
    def evaluation(self, 
                   prediction: str, 
                   answer: str,
                   **kwargs) -> dict:
        """
        Compute the evaluation metrics with model prediction and ground truth answer
        """