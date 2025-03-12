from .base import Algorithm


class Prompt(Algorithm):
    """
    single-step prompt:
    including direct prompt , few-shots in-context learning, chain-of-thought (CoT), 
    """
    def __init__(self, method='cot'):
        self.PROMPT_METHODS = ['zero-shot-cot', 'cot', 'l2m', ]
        self.method = method
        assert self.method in self.PROMPT_METHODS, f"Please choosing from {'; '.join(self.PROMPT_METHODS)}"

        self.prompt_name = method + '_prompt'
        # Initialize examples and question_template to avoid AttributeError
        self.examples = None
        self.question_template = None

    def do_reasoning(self, question, context=None) -> str:
        # if we use multiple examples, we need to transform list to a str
        if isinstance(self.examples, list):
            inputs = '\n'.join(self.examples) +'\n'
        elif self.examples:
            inputs = self.examples + '\n'
        else:
            inputs = ""

        if context:
            inputs += self.question_template.format(context=context, question=question)
        else:
            inputs += self.question_template.format(question=question)
        responds = self.model.generate([inputs]).text[0]
        return responds