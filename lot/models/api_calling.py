import json
import math
import os
import time
import warnings
from typing import Optional, Union

import numpy as np
from openai import OpenAI
try:
    from together import Together
except ImportError:
    print("together is not installed, please install it by `pip install togetherai`")

from .base import GenerateOutput, LanguageModel
from .utils import get_api_key, get_together_models

class opensource_API_models(LanguageModel):
    """
    A class for interacting with open source models via APIs like Together AI or local VLLM server.
    
    NOTE: For Together AI models, the model name should match those listed in 
    https://docs.together.ai/docs/inference-models
    """
    def __init__(self, 
        model:str, max_tokens:int=2048, temperature=0.0, additional_prompt=None, port=8000,
        system_prompt_type = 'CoT', local=False, local_api_key="token-abc123"
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.additional_prompt = additional_prompt
        
        # Determine which API to use based on the model name
        if local:
            self.api_key = local_api_key  # Dummy api_key for local server
            self.client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key=self.api_key)
        else:
            self.api_key = get_api_key("together")
            self.client = Together(api_key=self.api_key)

        # for rate limiting
        self.RATE_LIMIT = 40  # requests per minute; rate=40 cause 2min for Llama-3.1-70B-Instruct-Turbo
        self.INTERVAL = 60 / self.RATE_LIMIT  # interval in seconds between requests
        
        if system_prompt_type == 'default':
            self.SYSTEM_PROMPT = "You are a helpful assistant."
        elif system_prompt_type == 'CoT':
            self.SYSTEM_PROMPT = '''Your task is to provide answers using Chain of Thought (CoT) reasoning. This means breaking down your reasoning process into clear, independent steps. Each step should be written as a separate sentence. Do not combine multiple thoughts into one sentence. Instead, state each thought explicitly and independently.

    For example, if you are reasoning about the steps to solve a math problem:

    Start by identifying the given information.
    Determine the formula needed for the calculation.
    Substitute the given values into the formula.
    Perform the calculation to find the result.
    Check the result for accuracy.

    Please follow this format for all responses. Each thought or step should be clearly articulated in its own sentence.'''
        
    def generate(self,
                 prompt: Optional[Union[str, list[str]]],
                 max_tokens: int = None,
                 top_k: int = 0,
                 top_p: float = 1.0,
                 num_return_sequences: int = 1,
                 rate_limit_per_min: Optional[int] = None,
                 stop: Optional[str] = None,
                 logprobs: Optional[int] = None,
                 temperature = 1,
                 additional_prompt = None,
                 retry = 64,
                 use_chat_api = True,
                 **kwargs) -> GenerateOutput:
        
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        
        # check prompt
        inputs = prompt if isinstance(prompt, list) else [prompt]
        assert num_return_sequences > 0, 'num_return_sequences must be a positive value'
        if num_return_sequences > 1:
            assert len(inputs) == 1, 'num_return_sequences > 1 is not supported for multiple inputs'
        if additional_prompt is None and self.additional_prompt is not None:
            additional_prompt = self.additional_prompt
            
        elif additional_prompt is not None and self.additional_prompt is not None:
            warnings.warn("Warning: additional_prompt set in constructor is overridden.")

        responses, res_logprobs, output_tokens = [], [], []
        for each_prompt in inputs:
            success = 0
            for i in range(1, retry + 1):
                try:
                    # sleep several seconds to avoid rate limit
                    if rate_limit_per_min is not None:
                        time.sleep(60 / rate_limit_per_min)

                    if use_chat_api:
                        res = self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": self.SYSTEM_PROMPT},
                                {"role": "user", "content": each_prompt}
                            ],
                            max_tokens=max_tokens,
                            temperature=temperature,
                            # top_k=top_k,
                            top_p=top_p,
                            n=num_return_sequences,
                            stop=stop,
                            logprobs=1,
                        )        
                    else:
                        res = self.client.completions.create(
                            model=self.model,
                            prompt=each_prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            # top_k=top_k,
                            top_p=top_p,
                            n=num_return_sequences,
                            stop=stop,
                            logprobs=1,
                        )     
                    for i in range(num_return_sequences):
                        if use_chat_api:
                            responses.append(res.choices[i].message.content)
                        else:
                            responses.append(res.choices[i].text)                        
                        # res_logprobs.append(res.choices[i].logprobs.token_logprobs)
                        # output_tokens.append(res.choices[i].logprobs.tokens)
                    success = 1
                    break
                except Exception as e:
                    warnings.warn(f"An Error Occured: {e}, sleeping for {i} seconds")
                    time.sleep(i)
            if success == 0:
                raise RuntimeError(f"API request failed to generate output, even after {retry} tries")
            
        if logprobs == True:
            output = GenerateOutput(
                text=responses,
                log_prob=res_logprobs,
                str_each_token = output_tokens
            )
        else:
            output = GenerateOutput(
                text=responses
            )
        return output
    
    def get_loglikelihood(self,
                          prefix: str,
                          contents: list[str],
                          **kwargs) -> list[np.ndarray]:
        acc_probs = []
        for content in contents:
            # compute the logprob of generating the content (given the prefix)
            acc_probs.append(0)
            full_prompt = prefix + content
            res_echo = self.client.completions.create(model=self.model, prompt=full_prompt, max_tokens=1, logprobs=1, echo=True)
            cumulative_text = ""
            tokens = res_echo.prompt[0].logprobs.tokens
            for i in range(len(tokens)):
                token = tokens[i]
                cumulative_text += token
                if len(cumulative_text) > len(prefix):
                    logprobs = res_echo.prompt[0].logprobs.token_logprobs[i]
                    acc_probs[-1] += logprobs # accumulate the logprob
        return acc_probs

    def get_information_entropy(self,
                          prefix: str,
                          contents: list[str],
                          **kwargs) -> list[np.ndarray]:
        contents_entropy = []
        for content in contents:
            # compute the logprob of generating the content (given the prefix)
            full_prompt = prefix + content
            res_echo = self.client.completions.create(model=self.model, prompt=full_prompt, max_tokens=1, logprobs=1, echo=True)
            cumulative_text = ""
            tokens = res_echo.prompt[0].logprobs.tokens
            tokens_entropy = []
            for i in range(len(tokens)):
                token = tokens[i]
                cumulative_text += token
                if len(cumulative_text) > len(prefix):
                    logprobs = res_echo.prompt[0].logprobs.token_logprobs[i]
                    # with the probability p(y_i) of generating the token y_i
                    # entropy = -p(y_i) log(p(y_i))
                    entropy = -1 * math.exp(logprobs) * logprobs
                    tokens_entropy.append(entropy)
            contents_entropy.append(np.mean(tokens_entropy)) # np.sum?
        return contents_entropy
    
    def get_perplexity(self,
                        prefix: str,
                        contents: list[str],
                        **kwargs) -> list[np.ndarray]:
        
        contents_perplexity = []
        for content in contents:
            # compute the logprob of generating the content (given the prefix)
            full_prompt = prefix + content
            # together
            if isinstance(self.client, Together):
                res_echo = self.client.completions.create(model=self.model, prompt=full_prompt, max_tokens=1, logprobs=1, echo=True)
                cumulative_text = ""
                tokens = res_echo.prompt[0].logprobs.tokens
                tokens_logprobs = []
                for i in range(len(tokens)):
                    token = tokens[i]
                    cumulative_text += token
                    if len(cumulative_text) > len(prefix):
                        logprobs = res_echo.prompt[0].logprobs.token_logprobs[i]
                        tokens_logprobs.append(logprobs)
            # vllm
            elif isinstance(self.client, OpenAI):
                res_echo = self.client.completions.create(model=self.model, prompt=full_prompt, max_tokens=1, logprobs=1, echo=True)
                cumulative_text = ""
                tokens = res_echo.choices[0].logprobs.tokens
                tokens_logprobs = []
                for i in range(len(tokens)):
                    token = tokens[i]
                    cumulative_text += token
                    if len(cumulative_text) > len(prefix):
                        logprobs = res_echo.choices[0].logprobs.token_logprobs[i]
                        tokens_logprobs.append(logprobs)
            # calculate the perplexity
            avg_logprobs = np.mean(tokens_logprobs)
            perplexity = math.exp(-avg_logprobs)
            contents_perplexity.append(perplexity)
        return contents_perplexity
    
    def get_next_token_logits(self, 
                              prompt: str | list[str], 
                              candidates: list[str] | list[list[str]], 
                              postprocess: str | None = None, 
                              **kwargs) -> list[np.ndarray]:
        # assert postprocess in ['logits', 'logprobs', 'log_softmax', 'probs', 'softmax']
        prompts = [prompt] if isinstance(prompt, str) else prompt
        logprobs = []
        if isinstance(candidates[0], str): 
            candidates = [candidates] * len(prompt)
        for prompt, candidate_list in zip(prompts, candidates):
            logprob_list = []
            for candidate in candidate_list:
                full_prompt = prompt + candidate
                res_echo = self.client.completions.create(model=self.model, prompt=full_prompt, max_tokens=1, logprobs=1, echo=True)
                tokens = res_echo.prompt[0].logprobs.tokens
                last_token = tokens[-1]
                if candidate in last_token:
                    logprob_list.append(res_echo.prompt[0].logprobs.token_logprobs[-1])
                else:
                    matched_tokens = []
                    for i in range(-1, -len(tokens), -1):
                        matched_tokens.insert(0, tokens[i])
                        if candidate in ''.join(matched_tokens):
                            logprob_list.append(res_echo.prompt[0].logprobs.token_logprobs[i])
                            warnings.warn("warning: candidate {} has more than one token with {} tokens".format(candidate, -i))
                            break
            logprobs.append(logprob_list)
        return logprobs
    
# the following is used to test this script
if __name__ == '__main__':
    os.environ["TOGETHERAI_API_KEY"] = ""
    model = opensource_API_models(model="meta-llama/Meta-Llama-3-8B-Instruct-Lite", max_tokens=100)
    print(model.generate(['Hello, how are you?', 'How to go to Shanghai from Beijing?'], temperature= 0.7, top_p=0.7, top_k=50))
    print(model.get_loglikelihood("How can I goto Shanghai from Beijing?", [" By bicycle.", " By bus.", " By train.", " By air.", " By rocket.", " By spaceship."]))
    print(model.get_next_token_logits(["The capital of UK is", "The capital of France is", "The capital of Russia is"], ["Paris", "London", "Moscow"]))
    
"""
    REF:
    print(model.get_loglikelihood("How can I goto Shanghai from Beijing?", [" By bicycle.", " By bus.", " By train.", " By air.", " By rocket.", " By spaceship."]))
    # [-15.8808594, -11.998046800000001, -9.2929687, -10.4824218, -18.2714844, -20.1601562]
    print("*"*30)
    print(model.get_next_token_logits(["The capital of UK is", "The capital of France is", "The capital of Russia is"], ["Paris", "London", "Moscow"]))
    # [[-15.1328125, -8.0625, -16.109375], [-8.609375, -14.8671875, -17.90625], [-15.6171875, -15.34375, -7.1679688]]
    print("*"*30)
    print(model.get_loglikelihood("The capital of UK is", ["Paris", "London", "Moscow"]))
    # [-15.1328125, -8.0625, -16.128616333]
"""

