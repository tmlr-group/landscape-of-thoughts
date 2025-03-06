
import copy
import json
import os
import pickle as pkl

import numpy as np
from fire import Fire
from tqdm import tqdm

from algorithms import *
from models import *

"""
Trace of thoughts
Complexity: O(n) 
"""
    
def get_distance_matrix(model, model_input, answers, trial_thoughts, topk=10, default_distance=10, debug=False, asyn=False):
    """
    model: the language model;
    model_input: the input question;
    answers: the list of candidate answers;
    trial_thoughts: the list of chain of thoughts; each chain contains a list of split thoughts;
    
    The distance matrix is of size (num_all_thoughts+num_anchors, num_anchors)
    """
    
    # parse thoughts
    num_thoughts_each_chain = [len(thoughts) for [thoughts, _, _] in trial_thoughts]
    all_answers = [answer for [thoughts, answer, _] in trial_thoughts]
    all_thoughts = []
    for [thoughts, _, _] in trial_thoughts:
        all_thoughts += thoughts
    all_thoughts = np.array(all_thoughts)
    num_all_thoughts = len(all_thoughts)
    
    # parse anchors
    anchors = [model_input] + answers # question and answers
    num_anchors = len(anchors)
    anchors_idx_y = [i for i in range(num_anchors)]
    anchors_idx_x = [(num_all_thoughts + i) for i in range(num_anchors)]
    
    # initialize the distance matrix
    distance_matrix = np.ones((num_all_thoughts+num_anchors, num_anchors)) * default_distance
    
    # calculate the distance matrix
    for chain_idx in tqdm(range(len(trial_thoughts)), ncols=50):
        # prepare
        thoughts, _, _ = trial_thoughts[chain_idx]
        start_idx, end_idx = sum(num_thoughts_each_chain[:chain_idx]), sum(num_thoughts_each_chain[:chain_idx+1])
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

        # [1] compute p(state|question): X -> S_i
        #######################################
        perplexity_states_question = np.ones(len(states_without_question))*1 if debug else model.get_perplexity(model_input, states_without_question)
        distance_matrix[start_idx:end_idx, anchors_idx_y[0]] = np.array(perplexity_states_question)
        
        # [2] compute p(answer|state): S_i -> Y_1, Y_2, ..., Y_N
        #######################################
        for state_idx, state in enumerate(states_with_question):
            target_thoughts = copy.deepcopy(answers) # there is no next thought
            target_thoughts_idx = anchors_idx_y[1:]
            perplexity = np.ones(len(target_thoughts))*2 if debug else model.get_perplexity(state, target_thoughts)
            distance_matrix[start_idx+state_idx, target_thoughts_idx] = np.array(perplexity)
    
    # [3] get the anchors' coordinates
    # p(answer-1|question), p(answer-2|question), ..., p(answer-C|question) 
    #######################################
    perplexity = np.ones(len(answers))*3 if debug else model.get_perplexity(model_input, answers)
    distance_matrix[anchors_idx_x[0], anchors_idx_y[1:]] = np.array(perplexity) 
    distance_matrix[anchors_idx_x[1:], anchors_idx_y[0]] = np.array(perplexity)
    distance_matrix[anchors_idx_x[0], anchors_idx_y[0]] = 0 
    distance_matrix[anchors_idx_x[1:], anchors_idx_y[1:]] = 0 # make the diagonal be zeros
    
    return distance_matrix
    
def main(
    model_name: str = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
    port: int = 8000,
    thoughts_file: str = "None", 
    topk: int = 10,
    default_distance: int = 10,
    debug: bool = False,
    asyn: bool = False,
):

    # initialzie the language model
    #######################################    
    model = opensource_API_models(model=model_name, max_tokens=1000, port=port) # not stable outputs
    
    # load data
    ####################################### 
    trial_data = json.load(open(thoughts_file, 'r'))
    model_input = trial_data["model_input"]
    answers = trial_data["answers"]
    trial_thoughts = trial_data["trial_thoughts"]
    
    # compute distance matrix 
    #######################################
    save_path = copy.deepcopy(thoughts_file)
    save_path = save_path.replace(".json", f".pkl")
    save_path = save_path.replace("thoughts/", "distance_matrix/")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path) and not debug:
        print(f"==> distance matrix exists: {save_path}")
        exit()
    else:
        distance_matrix = get_distance_matrix(model, model_input, answers, trial_thoughts, 
                                              topk=topk, default_distance=default_distance, debug=debug, asyn=asyn)
        if not debug:
            print(f"==> save distance matrix to: {save_path}")
            pkl.dump(distance_matrix, open(save_path, 'wb'))
            
    print('==> the distance_matrix:\n\n', distance_matrix.shape, '\n', np.round(distance_matrix, 1))

if __name__ == "__main__":
    Fire(main)