import random

from ..tasks.science import SearchTask
from ..utils.solution_summary_extractor import extract_summary_from_solution
from ..utils.verify_MATH import exact_match_score
from .base import Node
from .bfs import BFS
from .dfs import DFS


class ToT_Task(SearchTask):
    def __init__(self, data, model=None, propose_method='glm', value_method='glm', algorithm='dfs', branch=3, select_branch=2,
                 max_depth=8, end_gate=0.9, select_method='greedy',
                 temperature=0.7, max_tokens=2048,
                 seed=170, max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=256, use_case_prompt=False, low=0, high=1, evaluate='', multiply_value=False, lang='zh', answer=None, verify_method='string', base_url="http://localhost:8000/v1"):
        super().__init__(data, propose_method, value_method)
        assert 0 <= low < high, "Inappropriate value range!"
        self.mode = 'tot'
        self.model = model # Using the trace-of-thought model instead of ReST-MCTS model
        self.base_url = base_url # for local model, it's default to be http://localhost:8000/v1

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.algorithm = algorithm
        self.branch = branch
        self.select_branch = select_branch
        self.max_depth = max_depth
        self.use_case_prompt = use_case_prompt
        self.low = low
        self.high = high
        self.evaluate = evaluate
        self.select_method = select_method
        self.end_gate = end_gate
        self.node_count = 1
        self.multiply_value = multiply_value
        self.lang = lang
        self.answer = answer
        self.verify_method = verify_method

    def update_count(self):
        self.node_count += 1

    def clear_cache(self):
        self.value_cache = {}
        self.node_count = 1

    def get_next_step(self, y, step_n):
        if self.use_case_prompt:
            prompt = self.single_propose_prompt_wrap(self.question, y, step_n)
        else:
            prompt = self.zero_single_propose_wrap(self.question, y, step_n, self.lang)

        response = self.model.generate(prompt).text[0]
        if not response:
            print('Failed to get next step！\n')
            return ''

        p = response.strip()

        if "Next step:" in p:
            stp = p.split('Next step:')[1].strip()
            if len(stp) < 2:
                print('The output step is too short！\n')
                return ''
            if stp in y:
                print('Output step repeat!\n')
                return ''

            revised_ = 'Step ' + str(step_n) + ': ' + stp
            print(f'New steps after standardization:{revised_}\n')
            return revised_ + '\n'

        elif "Step" in p and ":" in p:
            pre_len = len(p.split(':')[0])
            p_ = p[pre_len:]
            p_ = p_.split('Step')[0].strip()
            if len(p_) < 4:
                print('The output step is too short！\n')
                return ''
            p_ = p_[1:].strip()
            if p_ in y:
                print('Output step repeat!\n')
                return ''

            revised_ = 'Step ' + str(step_n) + ': ' + p_
            print(f'New steps after standardization:{revised_}\n')
            return revised_ + '\n'

        else:
            p_ = p.strip()
            if len(p_) < 3:
                print('The output step is too short！\n')
                return ''
            if p_ in y:
                print('Output step repeat!\n')
                return ''

            revised_ = 'Step ' + str(step_n) + ': ' + p_
            print(f'New steps after standardization:{revised_}\n')
            return revised_ + '\n'

    def get_step_value(self, y):
        if y in self.value_cache.keys():
            return self.value_cache[y]

        if self.value_method == 'local':
            prompt_answer = 'Problem: ' + self.question + '\nSolution:\n' + y

            value = self.model.generate(prompt_answer, temperature=self.temperature, max_tokens=self.max_tokens).text[0]
            print(f'Get a score:{value}\n')
            self.value_cache.update({y: value})
            return value

        else:
            prompt = self.value_prompt_wrap(self.question, y)
            response = self.model.generate(prompt, temperature=self.temperature, max_tokens=self.max_tokens).text[0]
            value = self.value_outputs_unwrap(response, self.low, self.high)
            print(f'Get a score:{value}\n')
            self.value_cache.update({y: value})
            return value

    def get_summary(self, y):

        prompt = self.MATH_summary_prompt_wrap(self.question, y)
        response = self.model.generate(prompt).text[0]
        summ = response.strip()

        print(f'Get summary:{summ}\n')
        return summ

    def run(self):
        self.clear_cache()
        if self.algorithm == 'dfs':
            solution, root, final_node = DFS(self)
        elif self.algorithm == 'bfs':
            solution, root, final_node = BFS(self)
        else:
            print('Unsupported algorithm!\n')
            return {}

        cnt = 5
        summary = ''
        while cnt:
            summary = self.get_summary(solution)
            if summary:
                break
            else:
                cnt -= 1
        if not summary and self.lang == 'en':
            summary = extract_summary_from_solution(solution)

        if self.evaluate == 'math' or self.verify_method == 'string':
            result = exact_match_score(summary, self.answer)
            final_answer = {'content': self.question, 'solution': solution, 'summary': summary, 'accurate': result, 'real_answer': self.answer}
        else:
            final_answer = {'content': self.question, 'solution': solution, 'summary': summary}

        if self.multiply_value:
            multiply_v = final_node.get_multiply_value()
            final_answer.update({'multiply_value': multiply_v})

        return final_answer, root
