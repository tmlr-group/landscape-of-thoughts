import os
import pickle
import sys
import numpy as np
import torch
from tqdm import tqdm
from transformers import StoppingCriteriaList
from abc import ABC, abstractmethod
from datetime import datetime
from typing import (Generic, NamedTuple, Optional, Protocol, Tuple, TypeVar,
                    Union, runtime_checkable)

class GenerateOutput(NamedTuple):
    text: list[str]
    log_prob: Optional[list[np.ndarray]] = None
    str_each_token: Optional[list[list[str]]] = None

class LanguageModel(ABC):
    @abstractmethod
    def generate(self,
                 inputs: list[str],
                 max_length: Optional[int] = None,
                 max_new_tokens: Optional[int] = None,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 num_return_sequences: int = 1,
                 eos_token_id: Union[None, str, int, list[str, int]] = None,
                 hide_input: bool = True,
                 output_log_probs: bool = False,
                 stopping_criteria: Optional[StoppingCriteriaList] = None,
                 use_chat_api = False,
                 **kwargs) -> GenerateOutput:
        """
        Generate text from a list of prompts.

        :param inputs: List of prompts.
        :param max_length: Maximum length of the total output (input + generated).
        :param max_new_tokens: Maximum length of generated tokens. Override max_length.
        :param do_sample: If False, do greedy decoding.
        :param temperature: Temperature for sampling.
        :param top_k: Top-k for sampling.
        :param top_p: Top-p for sampling.
        :param num_return_sequences:
        :param eos_token_id: Token id for end of sentence. Passed *str* will be translated into token_id.
                             Passed *list* will be treated as multiple possible tokens ending the generation.
        :param hide_input: If set true, decode only the generated part.
        :param output_log_probs: If set true, also output the log_probs of each generated token
        :param stopping_criteria
        :param use_chat_api: use chat or completions
        """
        ...
        
    @abstractmethod
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
        ...

    @abstractmethod
    def get_loglikelihood(self,
                          prefix: str,
                          contents: list[str],
                          **kwargs) -> np.ndarray:
        """Get the log likelihood of the contents given the prefix.

        :param prefix: The prefix to be excluded from the log likelihood.
        :param contents: The contents to evaluate (must include the prefix).
        """
        ...