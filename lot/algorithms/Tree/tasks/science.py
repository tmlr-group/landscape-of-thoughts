import os
import re

from .prompts import *


# data: question: str
# mode: 'cot', 'tot', 'mcts'
# method: 'glm', 'gpt', 'local'
class SearchTask(object):
    def __init__(self, data, propose_method='glm', value_method='glm'):
        super().__init__()
        self.question = data
        self.propose_method = propose_method
        self.value_method = value_method
        self.value_cache = {}

    def clear_cache(self):
        self.value_cache = {}

    @staticmethod
    def summary_prompt_wrap(x: str, y: str = '') -> str:
        print('\n', '==============================', 'summary', '==============================', '\n')
        print('summary_prompt: \n', x + '\nExisted Steps:\n' + y + 'Summary based on the existed steps:\n')
        prompt = summary_prompt + x + '\nExisted Steps:\n' + y + '\nOutput:'
        return prompt

    @staticmethod
    def MATH_summary_prompt_wrap(x: str, y: str = '') -> str:
        print('\n', '==============================', 'summary', '==============================', '\n')
        print('summary_prompt: \n', x + '\nExisted Steps:\n' + y + 'Summary based on the existed steps:\n')
        prompt = MATH_summary_prompt + x + '\nSolution: ' + y + '\nExtracted answer:'
        return prompt

    @staticmethod
    def evaluate_summary_prompt_wrap(x: str, y: str = '') -> str:
        print('\n', '==============================', 'summary', '==============================', '\n')
        print('summary_prompt: \n', x + '\nExisted Steps:\n' + y + 'Summary based on the existed steps:\n')
        prompt = evaluate_summary_prompt + x + '\nExisted Steps:\n' + y + '\nOutput:'
        return prompt

    @staticmethod
    def general_evaluate_summary_prompt_wrap(x: str, y: str = '') -> str:
        print('\n', '==============================', 'summary', '==============================', '\n')
        print('summary_prompt: \n', x + '\nExisted Steps:\n' + y + 'Summary based on the existed steps:\n')
        prompt = general_evaluate_summary_prompt + x + '\nExisted Steps:\n' + y + '\nOutput:'
        return prompt

    @staticmethod
    def single_propose_prompt_wrap(x: str, y: str = '', step: int = 0) -> str:
        print('\n', '==============================', 'proposal', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\nExisted Steps:\n' + y + 'Possible current step solution based on the existed steps:\n')
        prompt = single_proposal_prompt + x + '\nExisted Steps:\n' + y + '\nOutput:'
        return prompt

    @staticmethod
    def zero_single_propose_wrap(x: str, y: str = '', step: int = 0, lang: str = 'zh') -> str:
        print('\n', '==============================', 'proposal', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\nExisted Steps:\n' + y + 'Possible current step solution based on the existed steps:\n')
        if not y:
            y = 'None\n'
        prompt = zero_single_proposal_prompt_en + x + '\nExisting Steps:\n' + y + '\nOutput:'
        return prompt

    @staticmethod
    def zero_single_propose_wrap_mistral(x: str, y: str = '', step: int = 0) -> str:
        print('\n', '==============================', 'proposal', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\nExisted Steps:\n' + y + 'Possible current step solution based on the existed steps:\n')
        if not y:
            y = 'None\n'
        prompt = zero_single_proposal_prompt_mistral + x + '\nExisting Steps:\n' + y + '\nOutput:'
        return prompt

    @staticmethod
    def zero_single_propose_wrap_gpt(x: str, y: str = '', step: int = 0, lang: str = 'zh') -> str:
        print('\n', '==============================', 'proposal', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\nExisted Steps:\n' + y + 'Possible current step solution based on the existed steps:\n')
        if not y:
            y = 'None\n'
        prompt = zero_single_proposal_prompt_gpt_en + x + '\nExisting Steps:\n' + y + '\nOutput:'
        return prompt

    @staticmethod
    def zero_single_propose_wrap_use_reflection(x: str, y: str = '', step: int = 0, ref: str = '', lang: str = 'zh') -> str:
        print('\n', '==============================', 'proposal', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\nExisted Steps:\n' + y + 'Possible current step solution based on the existed steps:\n')
        if not y:
            y = 'None\n'
        if not ref:
            ref = 'None\n'
        prompt = zero_single_proposal_prompt_use_reflection_en + x + '\nExisting Steps:\n' + y + '\nAnalysis: ' + ref + '\nOutput:'
        return prompt

    @staticmethod
    def zero_single_propose_wrap_use_reflection_gpt(x: str, y: str = '', step: int = 0, ref: str = '', lang: str = 'zh') -> str:
        print('\n', '==============================', 'proposal', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\nExisted Steps:\n' + y + 'Possible current step solution based on the existed steps:\n')
        if not y:
            y = 'None\n'
        if not ref:
            ref = 'None\n'
        prompt = zero_single_proposal_prompt_use_reflection_gpt_en + x + '\nExisting Steps:\n' + y + '\nAnalysis: ' + ref + '\nOutput:'
        return prompt

    @staticmethod
    def single_reflection_wrap(x: str, y: str = '', step: int = 0, lang: str = 'zh') -> str:
        print('\n', '==============================', 'reflection', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\nExisted Steps:\n' + y + 'Based on the existed steps:\n')
        if not y:
            y = 'None\n'
        prompt = single_reflection_prompt_en + x + '\nExisting Steps:\n' + y + '\nOutput:'
        return prompt

    @staticmethod
    def single_reflection_wrap_gpt(x: str, y: str = '', step: int = 0) -> str:
        print('\n', '==============================', 'reflection', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\nExisted Steps:\n' + y + 'Based on the existed steps:\n')
        if not y:
            y = 'None\n'
        prompt = single_reflection_prompt_en + x + '\nExisting Steps:\n' + y + '\nOutput:'
        return prompt

    @staticmethod
    def single_reflection_wrap_llama(x: str, y: str = '', step: int = 0) -> str:
        print('\n', '==============================', 'reflection', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\nExisted Steps:\n' + y + 'Based on the existed steps:\n')
        if not y:
            y = 'None\n'
        prompt = single_reflection_prompt_en + x + '\nExisting Steps:\n' + y + '\nOutput:'
        return prompt

    @staticmethod
    def single_reflection_wrap_simple(x: str, y: str = '', step: int = 0, lang: str = 'zh') -> str:
        print('\n', '==============================', 'reflection', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\nExisted Steps:\n' + y + 'Based on the existed steps:\n')
        if not y:
            y = 'None\n'
        prompt = single_reflection_prompt_simple_en + x + '\nExisting Steps:\n' + y + '\nOutput:'
        return prompt

    @staticmethod
    def single_reflection_wrap_simple_mistral(x: str, y: str = '', step: int = 0) -> str:
        print('\n', '==============================', 'reflection', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\nExisted Steps:\n' + y + 'Based on the existed steps:\n')
        if not y:
            y = 'None\n'
        prompt = single_reflection_prompt_simple_mistral + x + '\nExisting Steps:\n' + y + '\nOutput:'
        return prompt

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        print('\n', '==============================', 'critic', '==============================', '\n')
        value_prompt = critic_simplified + x + '\nExisted Steps:\n' + y.strip() + '\nOutput:'
        return value_prompt

    @staticmethod
    def self_critic_prompt_wrap(x: str, y: str) -> str:
        print('\n', '==============================', 'self-critic', '==============================', '\n')
        if not y:
            y = 'None\n'
        critic_prompt = self_critic_prompt + x + '\nSolution:\n' + y + '\nScore:'
        return critic_prompt

    @staticmethod
    def cot_prompt_wrap(x: str, lang: str = 'zh', use_math: bool = False) -> str:
        print('\n', '==============================', 'proposal', '==============================', '\n')
        prompt = cot_prompt_en + x + "\nSolution:"
        print('propose_prompt: \n', prompt, '\n')
        return prompt

    @staticmethod
    def value_outputs_unwrap(value_outputs: list, low=0.0, high=1.0) -> float:
        out_value = low
        all_out = ''
        for _ in value_outputs:
            all_out = all_out + _
        all_out = all_out.lower()
        if 'score' not in all_out:
            print('Invalid score output!\n')
            return out_value
        stp = all_out.split('score')[-1].strip()
        try:
            match = re.findall(r'-?[0-9]+\.?[0-9]*', stp)[-1]
            out_value = float(match)
            out_value = min(max(low, out_value), high)
        except Exception as e:
            print(f'Invalid score output! Error type:{e}\n')
            return low
        return out_value
