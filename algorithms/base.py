from abc import ABC, abstractmethod

class Algorithm(ABC):
    
    def set_model(self, model):
        self.model = model
        
    def set_example(self, examples):
        self.examples = examples
        
    def set_question_template(self, question_template):
        self.question_template = question_template
        
    @abstractmethod  
    def do_reasoning(self, question) -> str:
        ...