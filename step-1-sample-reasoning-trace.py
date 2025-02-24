"""
Here is an example of the trial_data:

model_input (including the prompt and examples): 
Question: A car is being driven, in a straight line and at a uniform speed, towards the base of a vertical tower. The top of the tower is observed from the car and, in the process, it takes 10 minutes for the angle of elevation to change from 45° to 60°. After how much more time will this car reach the base of the tower? Answer Choices: (A) 5 (√3 + 1)  (B) 6 (√3 + √2)  (C) 7 (√3 – 1)  (D) 8 (√3 – 2)  (E) None of these. Answer:

answers (all the optional answers): 
['Answer is: A)5(√3 + 1)', 'Answer is: B)6(√3 + √2)', 'Answer is: C)7(√3 – 1)', 'Answer is: D)8(√3 – 2)', 'Answer is: E)None of these']

answer_gt_full (the right answer with the full information):
'Answer is: A)5(√3 + 1)'

answer_gt_short (the right answer in short):
'A'

trial_thoughts: 
[
    [["thought-1", "thought-2"], "A"], # trial-1, where A is the predicted answer
    [["thought-1", "thought-2"], "B"], # trial-2
    [...]
]
"""

import json
import os
import re
import time

from fire import Fire

from algorithms import MCTS_Task, ToT_Task
from algorithms.utils import get_tree
from models.opensource_API import opensource_API_models
from prompt import *

dataset2prompt = {
    "mmlu": prompt_mmlu,
    "strategyqa": prompt_strategyqa,
    "commonsenseqa": prompt_commonsenseqa,
    "aqua": prompt_aqua
}

dataset2pattern = {
    "mmlu": r'A|B|C|D',
    "strategyqa": r'A|B',
    "commonsenseqa": r'A|B|C|D',
    "aqua": r'A|B|C|D|E'
}

answer_idx_mapper = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4
    }

def main(
        model_name: str = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
        port: int = 8000,
        dataset_name: str = 'mmlu',
        dataset_path: str = 'data/mmlu_college_physics.json',
        method: str = 'cot',
        samples: int = 10,
        start_index: int = 0,
        end_index: int = 50
):
    print(f"==> model_name: {model_name}\n==> dataset_name: {dataset_name}\n==> dataset_path: {dataset_path}\n==> method: {method}\n==> samples_cnt: {samples}\n==> start_index: {start_index}\n==> end_index: {end_index}\n")

    # 1. Load the dataset
    _, ext = os.path.splitext(dataset_path)
    if ext not in ['.json', '.jsonl']:
        raise ValueError("file is not .json or .jsonl")

    # load JSON
    if ext == '.json':
        with open(dataset_path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
    # load JSONL
    elif ext == '.jsonl':
        dataset = []
        with open(dataset_path, 'r', encoding='utf-8') as file:
            for line in file:
                dataset.append(json.loads(line.strip()))

    # 2. parse the dataset
    if dataset_name == 'mmlu':
        questions = [item['question'] for item in dataset]
        options = [item['options'] for item in dataset]
        answers = [item['answer'] for item in dataset]
        rationales = None
    elif dataset_name == 'aqua':
        questions = [item['question'] for item in dataset]
        options = [item['options'] for item in dataset]
        answers = [item['correct'] for item in dataset]
        rationales = [item['rationale'] for item in dataset]
    elif dataset_name == 'commonsenseqa':
        questions = [item['question']['stem'] for item in dataset]
        options = [[item['question']['choices'][i]['label'] + ". " + item['question']['choices'][i]['text'] for i in range(5)] for item in dataset]
        answers = [item['answerKey'] for item in dataset]
        rationales = None
    elif dataset_name == 'strategyqa':
        questions = [item['question'] for item in dataset]
        options = [["A. yes", "B. no"] for item in dataset]
        answers = ["A" if item["answer"] else "B" for item in dataset]
        rationales = None
    else:
        raise ValueError("dataset_name is not supported")
    
    assert len(questions) == len(options) == len(answers)

    # 3. Initialize model
    model = opensource_API_models(model_name, max_tokens=2048, port=port)

    # 4. Generate thoughts
    print(f"==> start sampling ...")
    for i in range(start_index, end_index):
        print(f"==> sample: {i}/{end_index-start_index}")
        if method in ["tot", "mcts"]:
            save_root = "exp-data-searching"
        else:
            save_root = "exp-data-scale"
        save_path = f"{save_root}/{dataset_name}/thoughts/{model_name.split('/')[-1]}--{method}--{dataset_name}--{i}.json"
        if os.path.exists(save_path):
            print("==> skip: {}".format(save_path))
            continue

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        question, option, gt = questions[i], options[i], answers[i]
        query = question + "\nOptions: " + " ".join(option)        
        prompt = dataset2prompt[dataset_name][method].format(query=query)

        # form information
        answer_full_list = ["Answer is: " + opt for opt in option]
        answer_gt_full = answer_full_list[answer_idx_mapper[gt.strip()]]
        answer_gt_short = gt
        answer_gt_expl = ""
        if rationales:
            answer_gt_expl = rationales[i]

        trial_thoughts = []
        start_time = time.time()
        correct_cnt = 0
        for _ in range(samples):
            if method == 'mcts':
                task = MCTS_Task(prompt, model=model, propose_method='llama', value_method='llama',lang='en',iteration_limit=2)
                output, root = task.run()
                response = "\n".join(output[field] for field in ['content', 'solution','summary'])
            elif method == 'tot':
                task = ToT_Task(prompt, model=model, propose_method='llama', value_method='llama', algorithm='dfs', lang='en', max_depth=5)
                output, root = task.run()
                response = "\n".join(output[field] for field in ['content', 'solution','summary'])
            else:
                response = model.generate(prompt).text[0]

            matches = re.findall(pattern=dataset2pattern[dataset_name], string=response)
            pred = matches[-1] if matches else ""

            thoughts = [x.strip() for x in response.split("\n") if x.strip()]
            correctness = pred == gt
            if correctness:
                correct_cnt += 1
            trial_thoughts.append([thoughts, pred, correctness])
            
        
        end_time = time.time()
        print("==> time consuming: {:.2f}s".format(end_time - start_time))
        print("="*20)
        
        trial_data = {}
        trial_data["dataset"] = dataset_name
        trial_data["model"] = model_name
        trial_data["method"] = method
        trial_data["model_input"] = prompt
        trial_data["answers"] = answer_full_list
        trial_data["answer_gt_full"] = answer_gt_full
        trial_data["answer_gt_short"] = answer_gt_short
        trial_data["answer_gt_expl"] = answer_gt_expl
        trial_data["trial_thoughts"] = trial_thoughts
        trial_data["accuracy"] = correct_cnt / samples
        
        with open(save_path, 'w') as f:
            json.dump(trial_data, f)

        if method in ["tot", "mcts"]:
            df = get_tree(task, root)
            df_path = f"exp-data-searching/{dataset_name}/Tree/{model_name.split('/')[-1]}--{method}--{dataset_name}--{i}.json"
            if not os.path.exists(os.path.dirname(df_path)):
                os.makedirs(os.path.dirname(df_path))
            df.to_json(df_path)

if __name__ == "__main__":
    Fire(main)