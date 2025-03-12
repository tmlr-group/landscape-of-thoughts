from typing import Optional
from .base import BaseModel, LanguageModel, GenerateOutput

class ModelAdapter(BaseModel):
    """Adapter class to make LanguageModel implementations work as BaseModel."""
    
    def __init__(self, model: LanguageModel):
        self.model = model
    
    def generate(self, prompt: str, **kwargs) -> GenerateOutput:
        """Generate text from a prompt using the underlying LanguageModel."""
        # Convert single prompt to list as required by LanguageModel
        return self.model.generate([prompt], **kwargs)
    
    def get_likelihood(self, prompt: str, completion: str, **kwargs) -> float:
        """Get the likelihood of the completion given the prompt."""
        # Use the underlying model's get_loglikelihood method
        log_likelihood = self.model.get_loglikelihood(prompt, [completion], **kwargs)
        return float(log_likelihood[0])  # Convert first element to float 