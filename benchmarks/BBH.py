import copy
import json
import os
import pickle
import random
import sys
#import datasets as hugginfaceDatasets
from datasets import load_dataset
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import (Callable, Generic, NamedTuple, Optional, Protocol, Tuple, TypeVar, Union, runtime_checkable)

from .base import *
from .metrics import *




class bbh(Dataset):
    def __init__(self,
        example_root = './benchmarks/data/prompts/BBH',
        data_root = './benchmarks/data/BBH',
        output_extractor:Optional[Callable] = None,
        answer_extractor:Optional[Callable] = None,
        split:str = 'test',
        init_prompt:str = None,
        mode="FREE_FORM_TASKS",
        task_data="boolean_expressions",) -> None:

        self.example_root = example_root
        self.data_root = data_root
        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        self.input_processor = lambda x: x["question"]
        self.split = split
        self.task_data=task_data
        self.full_dataset = load_dataset(
            "lukaemon/bbh", task_data, split=self.split, cache_dir=f'{self.data_root}/{task_data}')
        # self.full_path=
        # self.full_dataset = hugginfaceDatasets.load_dataset(
        #      'json', data_files={'train': os.path.join(data_root, f'{self.task_data}.json')})
        self._dataset_name = 'BBH'
        
    def get_dataset(self, **kwargs) -> str:
        # Retrieves the dataset specified by name
        return self.full_dataset

    def get_example(self, shots=3, algorithm = 'cot_prompt'):
        # Retrieves the example specified by name
        with open(f"{self.example_root}/{self.task_data}.json") as f:
            examples = json.load(f)[algorithm]
            examples = examples.split('\n\n')
       
        if shots < len(examples):
            selected_samples = random.sample(examples, shots)
     
        return selected_samples
        
    def get_question_template(self):
        # formulate the input
        template = "Q: {question}\nA:"
        return template
    
    def is_number(self,s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def evaluation(self, prediction, answer):
        # Compute the evaluation metrics with model prediction and ground truth answer
        eval_results = {}
        print('answer',answer)
        if self.is_number(answer):
            eval_results['correctness'] = metric_correctness(prediction, answer)
        else:
            if prediction==answer:
                eval_results['correctness']=True
            else:
                eval_results['correctness']=False
        return eval_results
    