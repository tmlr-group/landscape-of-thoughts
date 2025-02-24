import json
import os
import pickle as pkl

import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.graph_objs as go
from scipy.interpolate import UnivariateSpline, interp1d, splev, splprep


def adjust_color(base_color, factor):
    """
    Create variations of the base color
    """
    if base_color == 'red':
        return f'rgb({255}, {int(100*factor)}, {int(100*factor)})'
    elif base_color == 'blue':
        return f'rgb({int(100*factor)}, {int(100*factor)}, {255})'
    return base_color

def normalize_chain_length(chains, num_points=50):
    """
    Normalize all chains to the same length while preserving the actual scale
    """
    normalized_chains = []
    
    for chain in chains:
        points = chain['points']
        if len(points) <= 1:
            continue
            
        # Calculate path length parameters
        path_dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        cum_dists = np.concatenate(([0], np.cumsum(path_dists)))
        
        # Use actual path distance instead of normalizing to [0,1]
        # This preserves the scale of the original points
        fx = interp1d(cum_dists, points[:, 0], bounds_error=False, fill_value="extrapolate")
        fy = interp1d(cum_dists, points[:, 1], bounds_error=False, fill_value="extrapolate")
        
        # Generate new points along the path
        new_dists = np.linspace(0, cum_dists[-1], num_points)
        new_points = np.column_stack((fx(new_dists), fy(new_dists)))
        normalized_chains.append(new_points)
    
    return np.array(normalized_chains)

def rearrange_columns(matrix, k):
    """
    Rearranges the columns of a given N x C matrix, 
    placing the k-th column first and shifting the others.
    
    Parameters:
    - matrix: a numpy array of shape (N, C).
    - k: the index (0-based) of the column to move to the first position.
    
    Returns:
    - A new numpy array with columns rearranged.
    
    This function is only used in load_data()
    """
    # Determine the new order of columns
    new_order = [k] + [i for i in range(matrix.shape[1]) if i != k]
    
    # Rearrange and return the new matrix
    return matrix[:, new_order]

def split_list(lengths, full_list):
    # Ensure the sum of lengths equals the length of the full list
    if sum(lengths) != len(full_list):
        raise ValueError("Sum of lengths must match the length of the full list")
    
    # Convert full list to a numpy array
    full_array = np.array(full_list)
    
    # Use numpy cumsum to generate indices for slicing
    indices = np.cumsum(lengths)[:-1]
    
    # Split the array at these indices
    split_arrays = np.split(full_array, indices)
    
    # Convert each split segment back to a list
    split_lists = [arr.tolist() for arr in split_arrays]
    
    return split_lists

def load_data(thoughts_file: str = "None"):
    # load thought file and parse
    #######################################
    assert os.path.exists(thoughts_file) 
    trial_data = json.load(open(thoughts_file, 'r'))
    answer_gt_short = trial_data["answer_gt_short"]
    trial_thoughts = trial_data["trial_thoughts"] # (thoughts, answer, binary_label) for each chain (i.e., each trial)
    pkl_path = thoughts_file.replace(".json", f".pkl")
    pkl_path = pkl_path.replace("thoughts/", "distance_matrix/")  
    assert os.path.exists(pkl_path)
    distance_matrix = pkl.load(open(pkl_path, 'rb'))

    # prepare the color of each chain for visualization
    #######################################
    chain_color = []
    for [_, answer, binary_label] in trial_thoughts:
        if binary_label == True:
            chain_color.append("green") # the final answer is correct
        else:
            chain_color.append("red") # the final answer is wrong
    
    # parse thoughts
    #######################################
    num_chains = len(trial_thoughts)
    num_thoughts_each_chain = [len(thoughts) for [thoughts, answer, binary_label] in trial_thoughts]
    all_answers = [answer for [thoughts, answer, binary_label] in trial_thoughts]
    all_thoughts = []
    for [thoughts, answer, binary_label] in trial_thoughts:
        all_thoughts += thoughts
    all_thoughts = np.array(all_thoughts)
    num_all_thoughts = len(all_thoughts)

    # draw the landscape
    #######################################    
    if "strategyqa" in thoughts_file:
        labels_anchors = ["Start", 'A', 'B']
        gt_idx = labels_anchors.index(answer_gt_short)
    elif "mmlu" in thoughts_file:
        labels_anchors = ["Start", 'A', 'B', 'C', 'D']
        gt_idx = labels_anchors.index(answer_gt_short)
    else:
        labels_anchors = ["Start", 'A', 'B', 'C', 'D', 'E']
        gt_idx = labels_anchors.index(answer_gt_short)
        
    # only use D(T, Y) for drawing (drop the first col)
    distance_matrix = distance_matrix[:num_all_thoughts+1, 1:] # get T matrix and the first row of the A matrix
    distance_matrix = distance_matrix / np.linalg.norm(distance_matrix, axis=1, ord=1, keepdims=True) # normalize the D (T, Y)
    
    # sort the source_matrix to make the GT at the first row (GT, Y_c, ... other answers)
    distance_matrix = rearrange_columns(distance_matrix, gt_idx-1)

    return distance_matrix, num_thoughts_each_chain, num_chains, num_all_thoughts, all_answers, answer_gt_short

def process_chain_points(chain_points):
    all_x = []
    all_y = []
    all_weights = []
    scatter_data = []
    start_points = []
    for chain in chain_points:
        points = chain['points']
        start_points.append(chain['start'])
        if len(points) <= 1:
            continue

        normalized_indices = np.linspace(0, 1, len(points))
        
        all_x.extend(points[:, 0])
        all_y.extend(points[:, 1])
        all_weights.extend(normalized_indices)

        scatter_data.append({
            'x': points[:, 0],
            'y': points[:, 1],
            'weights': normalized_indices
        })
    
    return all_x, all_y, all_weights, scatter_data, start_points
