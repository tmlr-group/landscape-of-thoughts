import json
import os
import pickle as pkl

import numpy as np
from scipy.interpolate import interp1d
from sklearn.manifold import TSNE
from tqdm import tqdm


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

def split_array(shapes, full_array):
    row_counts = [shape[0] for shape in shapes]
    assert sum(row_counts) == full_array.shape[0], "The total number of rows does not match"
    split_points = np.cumsum(row_counts)[:-1]
    return np.split(full_array, split_points)

def loading_data_from_file(model='Meta-Llama-3.1-70B-Instruct-Turbo', dataset='aqua', method="cot", total_sample=50, ROOT="./Landscape-Data"):
    # Load data
    ########################################
    plot_datas = {} 
    distance_matries = []
    num_all_thoughts_w_start_list = []

    for sample_idx in tqdm(range(total_sample), ncols=total_sample):
        # file_path = f'./exp-data-scale/{dataset}/thoughts/{model}--{method}--{dataset}--{sample_idx}.json'
        file_path = f'{ROOT}/{dataset}/thoughts/{model}--{method}--{dataset}--{sample_idx}.json'
        (distance_matrix, num_thoughts_each_chain, num_chains, num_all_thoughts, all_answers, answer_gt_short) = load_data(thoughts_file=file_path)
        
        plot_datas[sample_idx] = {
            "num_thoughts_each_chain": num_thoughts_each_chain,
            "num_chains": num_chains,
            "num_all_thoughts": num_all_thoughts,
            "all_answers": all_answers,
            "answer_gt_short": answer_gt_short
        }

        distance_matries.append(distance_matrix)
        num_all_thoughts_w_start_list.append(num_all_thoughts+1) # add one row from A matrix
    distance_matries = np.concatenate(distance_matries)
    return distance_matries, num_all_thoughts_w_start_list, plot_datas

def process_data(model='Meta-Llama-3.1-70B-Instruct-Turbo', dataset='aqua', method="cot", plot_type='method', total_sample=50, ROOT="./Landscape-Data", ):
    distance_matrix_shape = []
    list_distance_matrix = []
    list_num_all_thoughts_w_start_list = []
    list_plot_data = []

    if plot_type == "model":
        # assert method == 'cot', "model should be cot"
        # assert dataset == 'aqua', "dataset should be aqua"
        for model in ['Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct', 'Meta-Llama-3.1-8B-Instruct-Turbo', 'Meta-Llama-3.1-70B-Instruct-Turbo']:
            distance_matries, num_all_thoughts_w_start_list, plot_datas = loading_data_from_file(model=model, dataset=dataset, method=method, total_sample=total_sample, ROOT=ROOT)
            list_distance_matrix.append(distance_matries)
            list_plot_data.append(plot_datas)
            list_num_all_thoughts_w_start_list.append(num_all_thoughts_w_start_list)
            distance_matrix_shape.append(distance_matries.shape)
    
    elif plot_type == "dataset":
        # ! we cannot make all the sample with different num_answer to process together
        raise NotImplementedError
    
    elif plot_type == "method":
        # assert model == 'Meta-Llama-3.1-70B-Instruct-Turbo', "model should be 70B"
        # assert dataset == 'aqua', "dataset should be AQuA"
        for method in ['cot', 'l2m', 'mcts', 'tot']:
            distance_matries, num_all_thoughts_w_start_list, plot_datas = loading_data_from_file(model=model, dataset=dataset, method=method, total_sample=total_sample)
            list_distance_matrix.append(distance_matries)
            list_plot_data.append(plot_datas)
            list_num_all_thoughts_w_start_list.append(num_all_thoughts_w_start_list)
            distance_matrix_shape.append(distance_matries.shape)
    else:
        raise NotImplementedError

    fig_data = np.concatenate(list_distance_matrix)

    if dataset == "mmlu":
        target_A_matrix = np.ones((4,4)) * (1/4) 
    elif dataset == "strategyqa":
        target_A_matrix = np.ones((2,2)) * (1/3) 
    else:
        target_A_matrix = np.ones((5,5)) * (1/4) 
    target_A_matrix[np.diag_indices(target_A_matrix.shape[0])] = 0

    # concatenate all T and A(0-th row) (Nx(num_thoughts + 1), C), then concat the constant A matrix (C, C)
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    all_T_constant_A_distance_matrix = tsne.fit_transform(np.concatenate([fig_data, target_A_matrix]))

    # split the Nx(num_thoughts + 1) back to sample-wise distance matrix
    if dataset == "mmlu":
        index = -4
    elif dataset == "strategyqa":
        index = -2
    else:
        index = -5
    all_T_2D, A_matrix_2D = all_T_constant_A_distance_matrix[:index, :], all_T_constant_A_distance_matrix[index:, :]
    list_all_T_2D = split_array(distance_matrix_shape, all_T_2D)

    return list_all_T_2D, A_matrix_2D, list_plot_data, list_num_all_thoughts_w_start_list


def move_titles_to_bottom(fig, column_titles, y_position=-0.1, font_size=30):
    def update_annotation(a):
        if a.text in column_titles:
            a.update(y=y_position, font_size=font_size)
    fig.for_each_annotation(update_annotation)
    return fig