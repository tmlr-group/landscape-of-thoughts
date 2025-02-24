import copy
import json
import os
import pickle
import random
import sys
import datasets as hugginfaceDatasets
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import (Callable, Generic, NamedTuple, Optional, Protocol, Tuple, TypeVar, Union, runtime_checkable)

from .base import *
from .metrics import *

class Gsm8k(Dataset):
    def __init__(self,
        example_root = './benchmarks/data/prompts',
        data_root = './benchmarks/data',
        output_extractor:Optional[Callable] = None,
        answer_extractor:Optional[Callable] = None,
        split:str = 'test',
        init_prompt:str = None,) -> None:

        self.example_root = example_root
        self.data_root = data_root
        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        self.input_processor = lambda x: x["question"]
        self.split = split
        self.full_dataset = hugginfaceDatasets.load_dataset(
            'gsm8k', 'main', split=self.split, cache_dir=self.data_root+'/gsm8k')
        self._dataset_name = 'gsm8k'
        
    def get_dataset(self, **kwargs) -> str:
        # Retrieves the dataset specified by name
        return self.full_dataset

    def get_example(self, shots=3, algorithm = 'cot_prompt'):
        # Retrieves the example specified by name
        with open(f"{self.example_root}/gsm8k.json") as f:
            examples = json.load(f)[algorithm]

            # for zero-shot-cot, we using shots as prompt indexer
            if 'zero-shot-cot' in algorithm:
                examples = examples[shots % len(examples)]
            else:
                examples = examples.split('\n\n')
                if shots < len(examples):
                    examples = random.sample(examples, shots)
                    
        return examples
        
    def get_question_template(self):
        # formulate the input
        template = "Q: {question}\nA:"
        return template

    def evaluation(self, prediction, answer):
        # Compute the evaluation metrics with model prediction and ground truth answer
        eval_results = {}
        eval_results['correctness'] = metric_correctness(prediction, answer)
        return eval_results