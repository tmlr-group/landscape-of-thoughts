import os
import sys

import copy
import json
import pickle
import random
import argparse
import datasets as hugginfaceDatasets
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import (Callable, Generic, NamedTuple, Optional, Protocol, List, Tuple, TypeVar, Union, runtime_checkable)

from .base import *
from .metrics import *


# LongBench comprises 21 datasets across 6 task categories in both English and Chinese.
# LongBench has 6 tasks: singledoc QA, multi-doc QA, summarization, few-shot learning, synthetic tasks, and code completion.
class LongBench(Dataset):
    def __init__(self,
        example_root='./benchmarks/data/prompts',
        data_root='./benchmarks/data',
        output_extrantor: Optional[Callable]=None,
        answer_extractor: Optional[Callable]=None,
        split: str='test',
        init_prompt: str=None,
        dataset_name: str='narrativeqa') -> None:
        
        self.example_root = example_root
        self.data_root = data_root
        self.output_extrantor = output_extrantor
        self.answer_extractor = answer_extractor
        self.init_prompt = init_prompt
        self.input_processor = lambda dp: self.process_input(dp)
        self.split = split
        # TODO
        # LongBench和LongBench_e都支持进来
        self.datasets_list = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]  # all datasets in LongBench
        assert dataset_name in self.datasets_list
        self.dataset_name = dataset_name
        self.full_dataset = hugginfaceDatasets.load_dataset("THUDM/LongBench", self.dataset_name, split=self.split, \
                                                            cache_dir=self.data_root + f'/LongBench/{self.dataset_name}')
        self.dataset2metric = {
            "narrativeqa": qa_f1_score,
            "qasper": qa_f1_score,
            "multifieldqa_en": qa_f1_score,
            "multifieldqa_zh": qa_f1_zh_score,
            "hotpotqa": qa_f1_score,
            "2wikimqa": qa_f1_score,
            "musique": qa_f1_score,
            "dureader": rouge_zh_score,
            "gov_report": rouge_score,
            "qmsum": rouge_score,
            "multi_news": rouge_score,
            "vcsum": rouge_zh_score,
            "trec": classification_score,
            "triviaqa": qa_f1_score,
            "samsum": rouge_score,
            "lsht": classification_score,
            "passage_retrieval_en": retrieval_score,
            "passage_count": count_score,
            "passage_retrieval_zh": retrieval_zh_score,
            "lcc": code_sim_score,
            "repobench-p": code_sim_score,
        }
        self.metric = self.dataset2metric[self.dataset_name]
        self._dataset_name = f'LongBench-{self.dataset_name}'
    
    def get_dataset(self, **kwargs) -> str:
        return self.full_dataset
    
    def get_example(self, shots: int, algorithm: str, **kwargs) -> List[str]:
        # TODO
        pass

    def get_question_template(self):

        template_dict = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {question}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {question}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{question}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{question}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {question}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{question}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{question}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{question}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{question}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{question}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{question}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{question}Next line of code:\n"
}
        template = template_dict[self.dataset_name]
        return template

    def evaluation(self, prediction: str, answer: str, **kwargs) -> dict:  # TODO merge evaluation() and scorer() / scorer_e()
        eval_results = {}
        eval_results['score'] = self.metric(prediction=prediction, ground_truth=answer, kwargs=kwargs)
        return eval_results
    
    def process_input(self, dp: str) -> str:
        process_dict = {'input': dp['input'], 'context': dp['context']}
        return process_dict
    
    def parse_args(self, args=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, default=None)
        parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
        return parser.parse_args(args)

    def scorer_e(self, predictions, answers, lengths, all_classes):
        scores = {"0-4k": [], "4-8k": [], "8k+": []}
        for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
            score = 0.
            if self.dataset_name in ["trec", "triviaqa", "samsum", "lsht"]:
                prediction = prediction.lstrip('\n').split('\n')[0]
            for ground_truth in ground_truths:
                score = max(score, self.dataset2metric[self.dataset_name](prediction, ground_truth, all_classes=all_classes))
            if length < 4000:
                scores["0-4k"].append(score)
            elif length < 8000:
                scores["4-8k"].append(score)
            else:
                scores["8k+"].append(score)
        for key in scores.keys():
            scores[key] = round(100 * np.mean(scores[key]), 2)
        return scores

    def scorer(self, predictions, answers, all_classes):
        total_score = 0.
        for (prediction, ground_truths) in zip(predictions, answers):
            score = 0.
            if self.dataset_name in ["trec", "triviaqa", "samsum", "lsht"]:
                prediction = prediction.lstrip('\n').split('\n')[0]
            for ground_truth in ground_truths:
                score = max(score, self.dataset2metric[self.dataset_name](prediction, ground_truth, all_classes=all_classes))
            total_score += score
        return round(100 * total_score / len(predictions), 2)


# if __name__ == '__main__':
#     with open("benchmarks\data\prompts\game24.json") as f:
#         data = json.load(f)['cot_prompt']
#     import ipdb; ipdb.set_trace()

    # dataset = LongBench(example_root='D:/plusn/my_research/HKBU/reasoning-pro-dev/benchmarks/data/prompts',
    #                     data_root='D:/plusn/my_research/HKBU/reasoning-pro-dev/benchmarks/data',
    #                     split='test',
    #                     dataset_name='hotpotqa')
    
    