import copy
import json
import os
import pickle as pkl
import sys

import numpy as np
from fire import Fire
from tqdm import tqdm

from algorithms import *
from benchmarks import *
from models import *

"""
Trace of thoughts
Complexity: O(n) 
"""
    
def get_distance_matrix(model, model_input, trial_thoughts, default_distance=10,):
    """
    model: the language model;
    model_input: the input question;
    answers: the list of candidate answers;
    trial_thoughts: the list of chain of thoughts; each chain contains a list of split thoughts;
    
    The distance matrix is of size (num_all_thoughts+num_anchors, num_anchors)
    """
    
    # parse thoughts
    num_thoughts_each_chain = [len(thoughts) for [thoughts, _, _] in trial_thoughts]
    all_thoughts = []
    for [thoughts, _, _] in trial_thoughts:
        all_thoughts += thoughts
    all_thoughts = np.array(all_thoughts)
    
    # initialize the distance matrix
    # [num_chain, max_num_thought]
    distance_matrix = np.ones((len(trial_thoughts), max(num_thoughts_each_chain))) * default_distance
    # calculate the distance matrix
    for chain_idx in tqdm(range(len(trial_thoughts)), ncols=50):
        # prepare
        thoughts, _, _ = trial_thoughts[chain_idx]
        num_thoughts = num_thoughts_each_chain[chain_idx]
        
        # split the thoughts into states
        # note that the final answer (e.g., "Answer is: A") is included
        states_with_question = []
        states_without_question = []
        for i in range(len(thoughts)): 
            state_with_question = copy.deepcopy(model_input)
            state_without_question = ""
            for t in thoughts[:i+1]:
                state_with_question += " " + t
                state_without_question += " " + t
            states_with_question.append(state_with_question)
            states_without_question.append(state_without_question)
        assert len(states_with_question) == len(states_without_question)
        assert len(states_with_question) == num_thoughts

        # compute p(thought_i|state_i): S_0(Q) -> t_0 -> S_1
        # NOTE: each state would have the previous information. e.g., S_3 would include {S_0, t_0, S_1, t_1, S_2, t_2}
        #######################################
        for state_id in tqdm(range(num_thoughts), ncols=50):
            if state_id == 0:
                thought = thoughts[state_id]
                perplexity = model.get_perplexity(model_input, [thought])
            else:
                state = states_with_question[state_id-1]
                thought = thoughts[state_id]
                perplexity = model.get_perplexity(state, [thought])
            distance_matrix[chain_idx, state_id] = perplexity[0]
    return distance_matrix

def main(
    model_name: str = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
    port: int = 8000,
    thoughts_file: str = "None", 
    default_distance: int = 10,
):

    # initialzie the language model
    #######################################    
    model = opensource_API_models(model=model_name, max_tokens=1000, port=port) # not stable outputs
    
    # load data
    ####################################### 
    trial_data = json.load(open(thoughts_file, 'r'))
    model_input = trial_data["model_input"]
    trial_thoughts = trial_data["trial_thoughts"]
    
    # compute distance matrix 
    #######################################
    save_path = copy.deepcopy(thoughts_file)
    save_path = save_path.replace(".json", f".pkl")
    save_path = save_path.replace("thoughts/", "inter_distance_matrix/")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        print(f"==> distance matrix exists: {save_path}")
        exit()
    else:
        distance_matrix = get_distance_matrix(model, model_input, trial_thoughts, default_distance=default_distance)
        pkl.dump(distance_matrix, open(save_path, 'wb'))
            
    print('==> the distance_matrix:\n\n', distance_matrix.shape, '\n', np.round(distance_matrix, 1))
    
if __name__ == "__main__":
    Fire(main)