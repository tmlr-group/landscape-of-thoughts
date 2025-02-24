import os
import numpy as np
from typing import Optional, Union, Literal
import time
from litellm import completion
import warnings
from .base import GenerateOutput, LanguageModel

class closesource_API_models(LanguageModel):
    def __init__(self, model:str, max_tokens:int = 2048, temperature=0.0, additional_prompt=None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.additional_prompt = additional_prompt
    
    def generate(self,
                prompt: Optional[Union[str, list[str]]],
                max_tokens: int = None,
                top_p: float = 1.0,
                num_return_sequences: int = 1,
                rate_limit_per_min: Optional[int] = 60,
                stop: Optional[str] = None,
                logprobs: Optional[int] = None,
                temperature = None,
                additional_prompt=None,
                retry = 64,
                **kwargs) -> GenerateOutput:
        
        inputs = prompt if isinstance(prompt, list) else [prompt]
        assert num_return_sequences > 0, 'num_return_sequences must be a positive value'
        if num_return_sequences > 1:
            assert len(inputs) == 1, 'num_return_sequences > 1 is not supported for multiple inputs'
        
        # check hyper-parameters
        gpt_temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        logprobs = 0 if logprobs is None else logprobs
        
        if additional_prompt is None and self.additional_prompt is not None:
            additional_prompt = self.additional_prompt
        elif additional_prompt is not None and self.additional_prompt is not None:
            warnings.warn("Warning: additional_prompt set in constructor is overridden.")
        
        responses = []
        
        for each_prompt in inputs:
            success = 0
            for i in range(1, retry + 1):
                try:
                    # sleep several seconds to avoid rate limit
                    if rate_limit_per_min is not None:
                        time.sleep(60 / rate_limit_per_min)
                    
                    response = completion(
                        model=self.model, 
                        messages=[{"role": "user", "content": each_prompt}],
                        max_tokens=max_tokens,
                        temperature=gpt_temperature,
                        top_p=top_p,
                        stop=stop
                    )
                    
                    for i in range(num_return_sequences): 
                        responses.append(response.choices[i].message.content)
                    success = 1
                    break
                except Exception as e:
                    print(f"An Error Occured: {e}, sleeping for {i} seconds")
                    time.sleep(i)
            if success == 0:
                # after 64 tries, still no luck
                raise RuntimeError("GPTCompletionModel failed to generate output, even after 64 tries")
        return GenerateOutput(text=responses)
    
    
    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              postprocess: Optional[str] = None,
                              **kwargs) -> list[np.ndarray]:
        """ TODO: doc

        :param prompt:
        :param candidates:
        :param postprocess: optional, can be 'log_softmax' or 'softmax'. Apply the corresponding function to logits before returning
        :return:
        """
        warnings.warn("get_next_token_logits func is not supported in closesource api")
        pass
    
    def get_loglikelihood(self,
                          prefix: str,
                          contents: list[str],
                          **kwargs) -> np.ndarray:
        """Get the log likelihood of the contents given the prefix.

        :param prefix: The prefix to be excluded from the log likelihood.
        :param contents: The contents to evaluate (must include the prefix).
        """
        warnings.warn("get_loglikelihood func is not supported in closesource api")
        pass
    
if __name__ == '__main__':
    os.environ["GEMINI_API_KEY"] = ""
    model = closesource_API_models(model='gemini/gemini-pro')
    print(model.generate(['Hello, how are you?', 'How to go to Shanghai from Beijing?']).text)