import copy
import json
import os
import pickle as pkl
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import umap.umap_ as umap
from catboost import CatBoostClassifier
from einops import rearrange, reduce
from fire import Fire
from joblib import dump, load
from lightgbm import LGBMClassifier
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm, trange

MODEL_NAME_MAPPING = {
    "Llama-3.2-1B-Instruct": "Llama3_2-1B",
    "Llama-3.2-3B-Instruct": "Llama3_2-3B",
    "Meta-Llama-3.1-8B-Instruct-Turbo": "Llama3_1-8B",
    "Meta-Llama-3.1-70B-Instruct-Turbo": "Llama3_1-70B",
}

'''
    Utils
'''

def get_max_label(mode):
    if mode == 'cls': # for multi-cls classification
        return ['A', 'B', 'C', 'D', 'E']
    elif mode == 'reg': # for binary classification
        return [0, 1]

def pad_with_threshold(array, threshold=25, MAX_NUM_ANCHORS=5):
    if array.shape[0] > threshold:
        # Truncate to threshold length
        array = array[:threshold, :]
        pad_length = 0
        pad_anchor = MAX_NUM_ANCHORS - array.shape[1]
    else:
        # Pad to threshold length, and number of anchors
        pad_length = threshold - array.shape[0]
        pad_anchor = MAX_NUM_ANCHORS - array.shape[1]

    padding = ((0, pad_length), (0, pad_anchor))
    padded_array = np.pad(array, padding, mode='constant', constant_values=0)

    return padded_array

def save_models_and_metrics(randomforest_model, 
                          randomforest_X_scaler,
                          randomforest_y_scaler, 
                          llm_train_voting_acc, 
                          llm_train_avg_acc,
                          randomforest_mean_std,
                          ckpt_dir='ckpts'):
    
    # Create ckpts directory if it doesn't exist
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Save each object
    dump(randomforest_model, os.path.join(ckpt_dir, 'randomforest_model.pkl'))
    dump(randomforest_X_scaler, os.path.join(ckpt_dir, 'randomforest_X_scaler.pkl'))
    dump(randomforest_y_scaler, os.path.join(ckpt_dir, 'randomforest_y_scaler.pkl'))
    dump(llm_train_voting_acc, os.path.join(ckpt_dir, 'llm_train_voting_acc.pkl'))
    dump(llm_train_avg_acc, os.path.join(ckpt_dir, 'llm_train_avg_acc.pkl'))
    dump(randomforest_mean_std, os.path.join(ckpt_dir, 'randomforest_mean_std.pkl'))

def load_models_and_metrics(ckpt_dir='ckpts'):
    # Load each object
    randomforest_model = load(os.path.join(ckpt_dir, 'randomforest_model.pkl'))
    randomforest_X_scaler = load(os.path.join(ckpt_dir, 'randomforest_X_scaler.pkl'))
    randomforest_y_scaler = load(os.path.join(ckpt_dir, 'randomforest_y_scaler.pkl'))
    llm_train_voting_acc = load(os.path.join(ckpt_dir, 'llm_train_voting_acc.pkl'))
    llm_train_avg_acc = load(os.path.join(ckpt_dir, 'llm_train_avg_acc.pkl'))
    randomforest_mean_std = load(os.path.join(ckpt_dir, 'randomforest_mean_std.pkl'))
    
    return (randomforest_model, randomforest_X_scaler, 
            randomforest_y_scaler, llm_train_voting_acc, 
            llm_train_avg_acc, randomforest_mean_std)

'''
    Dataset Statistics info
'''
# LLM original Acc
def vote_accuracy(list_answers, list_answer_gt_short):
    voted_answers = [Counter(row).most_common(1)[0][0] if row else '' 
                    for row in list_answers]
    return sum(v == g for v, g in zip(voted_answers, list_answer_gt_short)) / len(list_answer_gt_short)

def row_wise_accuracy(list_answers, list_answer_gt_short):
    row_accs = [sum(ans == gt for ans in row if ans != '') / len([a for a in row if a != ''])
                if any(a != '' for a in row) else 0.0
                for row, gt in zip(list_answers, list_answer_gt_short)]
    return sum(row_accs) / len(row_accs)

'''
    Loading Datasets
'''

'''
    File --> Sample data
'''
def load_chain_data_and_plot(thoughts_file: str = "None",
    tool: str = 'tsne',
):

    # load data
    #######################################
    assert os.path.exists(thoughts_file), print(thoughts_file)
    trial_data = json.load(open(thoughts_file, 'r'))
    model_input = trial_data["model_input"]
    answers = trial_data["answers"]
    answer_gt_full = trial_data["answer_gt_full"]
    answer_gt_short = trial_data["answer_gt_short"]
    trial_thoughts = trial_data["trial_thoughts"]
    pkl_path = thoughts_file.replace(".json", f".pkl")
    pkl_path = pkl_path.replace("thoughts/", "distance_matrix/")  
    assert os.path.exists(pkl_path)
    distance_matrix = pkl.load(open(pkl_path, 'rb'))

    # pre-process for the visualize
    #######################################
    chain_color = []
    labels = []
    chain_corr = []
    for [_, answer, binary_label] in trial_thoughts:
        if binary_label == True:
            chain_color.append("green")
            chain_corr.append(binary_label)
        else:
            chain_color.append("red")
            chain_corr.append(binary_label)
        labels.append(binary_label)

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
    anchors = [model_input] + answers # question and answers
    num_anchors = len(anchors)
    anchors_idx_y = [i for i in range(num_anchors)]
    anchors_idx_x = [(num_all_thoughts + i) for i in range(num_anchors)]

    # draw the landscape
    #######################################    
    if "strategyqa" in thoughts_file:
        labels_anchors = ["Start", 'A', 'B']
        if answer_gt_short:
            answer_gt_short = 'A' # yes
        else:
            answer_gt_short = 'B' # no
        gt_idx = labels_anchors.index(answer_gt_short)
    else:
        labels_anchors = ["Start", 'A', 'B', 'C', 'D', 'E']
        gt_idx = labels_anchors.index(answer_gt_short)
    answer_idx_y = anchors_idx_y[gt_idx] # the groud truth answer

    # # normlize thought (T matrix)  
    processed_distance_matrix = copy.deepcopy(distance_matrix)  
    # ######################################################
    
    for chain_idx in range(num_chains): 
        start_idx, end_idx = sum(num_thoughts_each_chain[:chain_idx]), sum(num_thoughts_each_chain[:chain_idx+1])
        accumulate_ppl = 0
        for thought_idx in range(start_idx, end_idx):
            accumulate_ppl += distance_matrix[thought_idx, 0]
            # norm D(X, T)
            processed_distance_matrix[thought_idx, 0] = accumulate_ppl / np.sum(distance_matrix[start_idx:end_idx, 0])
            # norm D(T, Y)
            for anchor_idx in range(1, num_anchors):
                processed_distance_matrix[thought_idx, anchor_idx] = distance_matrix[thought_idx, anchor_idx] / np.sum(distance_matrix[thought_idx, anchors_idx_y[1:]])

    # check the normalize effect
    for chain_idx in range(num_chains): 
        start_idx, end_idx = sum(num_thoughts_each_chain[:chain_idx]), sum(num_thoughts_each_chain[:chain_idx+1])
        # assert np.abs(np.sum(processed_distance_matrix[start_idx:end_idx, 0])- 1) < 1e-5 # first col, each chian sum would be 1
        for thought_idx in range(start_idx, end_idx): # row-sum=1
            assert ( (np.sum(processed_distance_matrix[start_idx:end_idx, anchors_idx_y[1:]], axis=1) - 1) < 1e-5 ).all()

    # normlize answer (A matrix)
    ######################################################
    A = processed_distance_matrix[num_all_thoughts:]
    A[np.diag_indices(A.shape[0])] = 0
    normed_A = copy.deepcopy(A)
    for col_idx in range(1, num_anchors):
        normed_A[0][col_idx] = 1 + A[0][col_idx] / np.sum(A[0, anchors_idx_y[1:]])
    normed_A[:, 0] = normed_A[0, :] # copy same elements to 0-th col
    normed_A[1:, 1:] = 1 / (num_anchors-1) 
    normed_A[np.diag_indices(normed_A.shape[0])] = 0
    processed_distance_matrix[num_all_thoughts:] = normed_A

    distance_matrix = processed_distance_matrix

    if tool == 'tsne':
        tsne = TSNE(n_components=2, perplexity=10, random_state=42)
        coordinates_2d = tsne.fit_transform(distance_matrix[:, 1:])
    elif tool == 'umap':
        coordinates_2d = umap.UMAP(n_neighbors=30, min_dist=0.25, n_components=2, metric='dice', random_state=42).fit_transform(distance_matrix)

    return distance_matrix, num_chains, num_thoughts_each_chain, coordinates_2d, normed_A, all_answers, answer_gt_short, coordinates_2d[anchors_idx_x[gt_idx], :]

'''
    Sample data --> List of Chain data
'''
def load_sample_data(
    model='Meta-Llama-3.1-70B-Instruct-Turbo',
    dataset='aqua',
    method="zero_shot_cot",
    tool='tsne',
    root='./training_data',
    start_sample_idx=0,
    end_sample_idx=10
):
    list_distance_matrix = []
    list_num_chains = []
    list_num_thoughts_each_chain = []
    list_coordinates_2d = []
    list_normed_A = []
    list_answers = []
    list_answer_gt_short = []
    list_answer_gt_coordinates_2d = []
    print(f"==> Loading {model}--{method}--{dataset}...")
    for sample_idx in range(start_sample_idx, end_sample_idx):
        file_path = f'{root}/{dataset}/thoughts/{model}--{method}--{dataset}--{sample_idx}.json'
        
        try:
            (
                distance_matrix, num_chains, num_thoughts_each_chain, coordinates_2d, 
                normed_A, answers, answer_gt_short, answer_gt_coordinates_2d
            ) = load_chain_data_and_plot(thoughts_file=file_path, tool=tool)

            list_distance_matrix.append(distance_matrix)
            list_num_chains.append(num_chains)
            list_num_thoughts_each_chain.append(num_thoughts_each_chain)
            list_coordinates_2d.append(coordinates_2d)
            list_normed_A.append(normed_A)
            list_answers.append(answers)
            list_answer_gt_short.append(answer_gt_short)
            list_answer_gt_coordinates_2d.append(answer_gt_coordinates_2d)
        except:
            print(f'Missing {file_path}')
            continue

    return (
        list_distance_matrix, list_num_chains, list_num_thoughts_each_chain,
        list_coordinates_2d, list_answers, list_answer_gt_short, list_normed_A
    )

'''
    List of Chain data --> List of (chain distance matrix, label)
'''
def preprocess_data(
    list_distance_matrix, list_num_chains,
    list_num_thoughts_each_chain, list_coordinates_2d,
    list_answers, list_answer_gt_short,
    mode="cls"
):
    training_data_sample_2d = []
    training_data_sample_matrix = []

    for sample_idx in range(len(list_answers)):
        num_chains = list_num_chains[sample_idx]
        num_thoughts_each_chain = list_num_thoughts_each_chain[sample_idx]
        coordinates_2d = list_coordinates_2d[sample_idx]
        distance_matrix = list_distance_matrix[sample_idx]
        model_answers = list_answers[sample_idx]
        gt_answer = list_answer_gt_short[sample_idx]
        
        training_data_chain_2d = []
        training_data_chain_matrix = []

        for chain_idx in range(num_chains):
            start_idx = sum(num_thoughts_each_chain[:chain_idx])
            end_idx = sum(num_thoughts_each_chain[:chain_idx+1])
            chain_coordinates_2d = coordinates_2d[start_idx:end_idx, :]
            chain_matrix = distance_matrix[start_idx:end_idx, :]

            if mode == "cls":
                training_data_chain_2d.append((chain_coordinates_2d, gt_answer))
                training_data_chain_matrix.append((chain_matrix, gt_answer))
            elif mode == "reg":
                chain_answer = model_answers[chain_idx] if model_answers[chain_idx] else "A"
                training_data_chain_2d.append((chain_coordinates_2d, gt_answer==chain_answer))
                training_data_chain_matrix.append((chain_matrix, gt_answer==chain_answer))
            else:
                raise NotImplementedError

        training_data_sample_2d.append(training_data_chain_2d)
        training_data_sample_matrix.append(training_data_chain_matrix)

    return training_data_sample_2d, training_data_sample_matrix


def collect_valid_coordinates_and_answers(training_data_sample, dataset, model_type='lgb'):
    """
    Process training data sample to collect valid coordinates and answers
    
    Args:
        training_data_sample: A list of chains, where each chain contains tuples of (coordinates, answer)
    Returns:
        combined_coords: numpy array of coordinates
        answers_array: numpy array of corresponding answers
    """
    all_coords = []
    all_answers = []

    try:
        for chain_idx, chain in enumerate(training_data_sample):
            # Check if chain is a tuple (should be (coordinates, answer))
            if isinstance(chain, tuple) and len(chain) == 2:
                coordinates, answer = chain
                if not isinstance(coordinates, (np.ndarray, list)) or not len(coordinates):
                    print(f"Empty coordinates at chain {chain_idx}")
                    continue

                try:
                    coords_array = np.array(coordinates, dtype=np.float64)
                    if coords_array.size == 0 or np.any(np.isnan(coords_array)):
                        print(f"Invalid coordinates at chain {chain_idx}")
                        continue

                    # remove T(X, Y) if exised
                    if dataset == "strategyqa":
                        if coords_array.shape[1] == 3:
                            coords_array = coords_array[:, 1:] 
                    elif dataset == "mmlu":
                        if coords_array.shape[1] == 5:
                            coords_array = coords_array[:, 1:] 
                    else:
                        if coords_array.shape[1] == 6:
                            coords_array = coords_array[:, 1:] 
    
                    if coords_array.shape[0] == 1: continue # remove the chain with only one thought

                    # this ensure the coords_array as (N, num_anchors)
                    coords_array = pad_with_threshold(coords_array, threshold=30) # ! truncate/pad the number of thought
                    all_coords.append(coords_array) # ! make sure all the coordinates are in the same shape (N, D)
                    all_answers.append(answer)

                except (ValueError, TypeError) as e:
                    print(f"Error processing coordinates at chain {chain_idx}: {e}")
                    continue
            else:
                print(f"Invalid chain format at index {chain_idx}")
                continue

    except Exception as e:
        print(f"Error in data processing: {e}")
        return None, None

    if not all_coords:
        print("No valid coordinates collected")
        return None, None

    try:
        # TODO: use RNN/LSTM, so we can directly use (n, l, d) as input
        combined_coords = np.array(all_coords) 
        # # NOTE: this will be (N, 30*5), N is the number of samples
        if model_type == 'lgb':
            combined_coords = rearrange(combined_coords, 'n l d -> n (l d)') 
        # # using max/mean/sum pooling
        # X_mean = reduce(combined_coords, 'n l d -> n d', 'mean')  # Reduce length dimension with mean
        # X_max = reduce(combined_coords, 'n l d -> n d', 'max')    # Reduce length dim/ension with max
        # X_sum = reduce(combined_coords, 'n l d -> n d', 'sum')    # Reduce length dimension with sum
        # # Combine features
        # # # NOTE: this will be (N, 3*5), N is the number of samples
        # combined_coords = np.hstack([X_mean, X_max, X_sum]) # n, 3*d

        answers_array = np.array(all_answers)
        return combined_coords, answers_array

    except Exception as e:
        for coord in all_coords:
            print(f"Coordinates: {coord.shape}")
        print(f"Error during final processing: {e}")
        return None, None

'''
    Multiple config datasets --> X, y, meta_info
'''
def prepare_data_for_training(dataset_configs, verbose=False, mode='reg', model_type='lgb'):
    """
    Prepare data for training including original accuracy calculations
    """
    all_data = []
    meta_info = []
    label_sets = set()
    voting_accs = []
    average_accs = []
    
    # Loading Raw data
    ##########################################
    for config in dataset_configs:
        list_distance_matrix, list_num_chains, list_num_thoughts_each_chain, \
        list_coordinates_2d, list_answers, list_answer_gt_short, list_normed_A = \
            load_sample_data(
                model=config['model'],
                dataset=config['dataset'],
                method=config['method'],
                start_sample_idx=config['start_idx'],
                end_sample_idx=config['end_idx']
            )
        
        # Calculate original accuracies
        voting_acc = vote_accuracy(list_answers, list_answer_gt_short)
        average_acc = row_wise_accuracy(list_answers, list_answer_gt_short)
        
        voting_accs.append(voting_acc)
        average_accs.append(average_acc)
        
        # Process data
        _, training_data_matrix = preprocess_data(
            list_distance_matrix, list_num_chains,
            list_num_thoughts_each_chain, list_coordinates_2d,
            list_answers, list_answer_gt_short,
            mode=mode,
        )

        # NOTE: we use the full distance matrix for training
        for sample_data in training_data_matrix:
            X, y = collect_valid_coordinates_and_answers(sample_data, dataset=config['dataset'], model_type=model_type)
            if X is not None and y is not None:
                current_labels = np.unique(y)

                
                if not set(current_labels).issubset(set([0, 1,])): 
                    if verbose:
                        print(f"Warning: Dataset {config['dataset']} contains invalid labels: {set(current_labels) - set(['A', 'B', 'C', 'D', 'E'])}")
                    continue

                label_sets.update(current_labels)
                all_data.append((X, y)) # chains X: (10, D); y: (10, )
                meta_info.extend([{
                    'dataset': config['dataset'],
                    'model': config['model'],
                    'method': config['method'],
                }] * len(X))

    ##########################################
    # Combine all data
    # First, check dimensions
    X_list = []
    y_list = []

    for X, y in all_data:
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y, list):
            y = np.array(y)
        X_list.append(X)
        y_list.append(y)

    # Concatenate along the first dimension (number of anchors)
    X_combined = np.concatenate(X_list, axis=0)
    y_combined = np.concatenate(y_list, axis=0)

    # Convert meta information
    meta_df = pd.DataFrame(meta_info)
    meta_encoded = pd.get_dummies(meta_df, columns=['dataset', 'model', 'method'])
    # Combine features with meta information
    if model_type == 'lgb':
        X_with_meta = np.hstack([X_combined, meta_encoded.values]) # (num_chains, 5*50+meta_infos)
    else:
        meta_expanded = np.repeat(meta_encoded.values[:, np.newaxis, :], X.shape[1], axis=1)
        X_with_meta = np.concatenate((X_combined, meta_expanded), axis=2)

    # Prepare for training
    label_encoder = LabelEncoder()
    label_encoder.fit(get_max_label(mode=mode))
    
    if model_type == 'lgb':
        scaler = StandardScaler()
        X_scaled = X_with_meta.copy()
        X_scaled = scaler.fit_transform(X_with_meta)
    else:
        # original_shape = X_with_meta.shape
        # X_with_meta_flattened = X_with_meta.reshape(original_shape[0], -1)
        # scaler = StandardScaler()
        # X_scaled_flattened = X_with_meta_flattened.copy()
        # X_scaled_flattened = scaler.fit_transform(X_with_meta_flattened)
        # X_scaled = X_scaled_flattened.reshape(original_shape)

        # b: batch, s: sequence length, f: features

        # Method 1: Using BatchNorm1d with einops
        X_scaled = torch.tensor(X_with_meta.copy(), device='cpu', dtype=torch.float32)
        scaler = torch.nn.BatchNorm1d(X_scaled.shape[-1])
        X_scaled = rearrange(X_scaled, 'batch seq_len dim -> batch dim seq_len')  # (6980, 15, 30)
        X_scaled = scaler(X_scaled)
        X_scaled = rearrange(X_scaled, 'batch dim seq_len -> batch seq_len dim')  # (6980, 30, 15)
    y_scaled = label_encoder.transform(y_combined)
    
    if verbose:
        print("\nData Processing Summary:")
        print(f"Total samples: {len(y_combined)}")
        print(f"Feature dimensionality: {X_with_meta.shape[1]}")
        print(f"Number of unique labels: {len(label_sets)}")
        print(f"Labels present: {sorted(list(label_sets))}")
        print(f"\nAverage LLM Performance:")
        print(f"Average Voting Accuracy: {np.mean(voting_accs):.2f}")
        print(f"Average Chain Accuracy: {np.mean(average_accs):.2f}")
    

    acc_infos = {
        'voting_accs': voting_accs,
        'average_accs': average_accs,
        'mean_voting_acc': np.mean(voting_accs),
        'mean_average_acc': np.mean(average_accs)
    }

    return (X_scaled, y_scaled, scaler, label_encoder, acc_infos)

'''
    Evaluation
'''

'''
    List of Chain data --> (chain distance matrix, label)

    NOTE: This is similar to the `preprocess_data`, 
    but loading single sample data, instead of all samples.
'''

def load_chain_data(
        sample_idx,
        list_num_chains,
        list_num_thoughts_each_chain,
        list_coordinates_2d,
        list_distance_matrix,
        list_answers,
        list_answer_gt_short,
        mode="cls"
    ):
    num_chains = list_num_chains[sample_idx]
    num_thoughts_each_chain = list_num_thoughts_each_chain[sample_idx]
    coordinates_2d = list_coordinates_2d[sample_idx]
    distance_matrix = list_distance_matrix[sample_idx]
    model_answers = list_answers[sample_idx]
    gt_answer = list_answer_gt_short[sample_idx]

    training_data_chain_2d = []
    training_data_chain_matrix = []
    for chain_idx in range(num_chains):
        start_idx, end_idx = sum(num_thoughts_each_chain[:chain_idx]), sum(num_thoughts_each_chain[:chain_idx+1])
        chain_coordinates_2d = coordinates_2d[start_idx:end_idx, :]
        chain_matrix = distance_matrix[start_idx:end_idx, :]
        if mode == "cls":
            training_data_chain_2d.append((chain_coordinates_2d, gt_answer)) # for classification
            training_data_chain_matrix.append((chain_matrix, gt_answer))
        elif mode == "reg":
            chain_answer = model_answers[chain_idx] if model_answers[chain_idx] else "A" # default answer for the chain with empty values
            training_data_chain_2d.append((chain_coordinates_2d, gt_answer==chain_answer)) # for regression
            training_data_chain_matrix.append((chain_matrix, gt_answer==chain_answer)) # for regression
        else:
            raise NotImplementedError
        
    return training_data_chain_2d, training_data_chain_matrix

def chain_collect_valid_coords_and_answers(list_chain_coord_answers, dataset, model_type='lgb'):
    all_coords = []
    all_answers = []
    for chain_idx, (coordinates, answer) in enumerate(list_chain_coord_answers):
        if not len(coordinates):
            print(f"Empty coordinates at chain {chain_idx}")
            continue
            
        try:
            coords_array = np.array(coordinates, dtype=np.float64)
            
            if coords_array.size == 0 or np.any(np.isnan(coords_array)):
                print(f"Invalid coordinates at chain {chain_idx}")
                continue
                
            # remove T(X, Y) if exised
            if dataset == "strategyqa":
                if coords_array.shape[1] == 3:
                    coords_array = coords_array[:, 1:] 
            elif dataset == "mmlu":
                if coords_array.shape[1] == 5:
                    coords_array = coords_array[:, 1:] 
            else:
                if coords_array.shape[1] == 6:
                    coords_array = coords_array[:, 1:] 
            
            # truncate/pad the number of thought, with maximun of threshold (default: 50)
            # this ensure the coords_array as (N, 5)
            coords_array = pad_with_threshold(coords_array, threshold=30) 
            
            # make sure all the coordinates are in the same shape (N, D)
            all_coords.append(coords_array) 
            all_answers.append(answer)

        except (ValueError, TypeError) as e:
            print(f"Error processing coordinates at chain {chain_idx}: {e}")
            continue

    if not all_coords:
        print("No valid coordinates collected")
        return None, None

    try:
        combined_coords = np.array(all_coords) 
        # # NOTE: this will be (N, 30*5), N is the number of samples
        if model_type == 'lgb':
            combined_coords = rearrange(combined_coords, 'n l d -> n (l d)')
        # using max/mean/sum pooling
        # X_mean = reduce(combined_coords, 'n l d -> n d', 'mean')  # Reduce length dimension with mean
        # X_max = reduce(combined_coords, 'n l d -> n d', 'max')    # Reduce length dim/ension with max
        # X_sum = reduce(combined_coords, 'n l d -> n d', 'sum')    # Reduce length dimension with sum
        # Combine features
        # # NOTE: this will be (N, 3*5), N is the number of samples
        # combined_coords = np.hstack([X_mean, X_max, X_sum]) # n, 3*d

        answers_array = np.array(all_answers)

        # if mode == "cls":
        #     answers_array = np.unique(np.array(all_answers))
        #     assert len(answers_array) == 1, "Answers are not consistent within the chain"

        return combined_coords, answers_array

    except Exception as e:
        print(f"Error during final processing: {e}")
        return None, None

def calculate_accuracies(y_preds, y_true, model_pred_mode='reg'):
    # Voting
    if model_pred_mode == 'cls':
        vote_acc = 1 if mode(y_preds)[0] == y_true else 0
    else:
        vote_acc = -1
    # average accuracy
    correct_predictions = np.sum(y_preds == y_true)
    avg_acc = correct_predictions / len(y_preds)

    return vote_acc, avg_acc

def create_consistent_meta_encoding(config, X_length, original_columns):
    """
    创建与训练数据一致的meta编码
    
    Args:
        config: 包含 dataset, model, method 的配置
        X_length: 样本数量
        original_columns: 原始训练数据的编码列名列表
    """
    # 创建单个样本的meta信息
    meta_info = [{
        'dataset': config['dataset'],
        'model': config['model'],
        'method': config['method'],
    }] * X_length
    
    # 转换为DataFrame
    meta_df = pd.DataFrame(meta_info)
    
    # 使用相同的列进行one-hot编码
    meta_encoded = pd.get_dummies(meta_df, columns=['dataset', 'model', 'method'])
    
    # 确保所有原始列都存在，没有的填0
    for col in original_columns:
        if col not in meta_encoded.columns:
            meta_encoded[col] = 0
            
    # 确保列的顺序一致
    meta_encoded = meta_encoded[original_columns]
    
    return meta_encoded.values

'''
    Random Forest Evaluation
'''

def eval_random_forest(
        data: list = None, 
        meta_info: dict = None,
        x_scaler: StandardScaler = None,
        y_scaler: StandardScaler = None,
        models: list = [LGBMClassifier],
        model_pred_mode: str = 'cls',
        model_type: str = 'lgb',
    ):
    # X: (num_chains, 50*5+meta_info_dim)
    # y: (num_chains, )
    X, y = chain_collect_valid_coords_and_answers(data, dataset=meta_info['dataset'], model_type=model_type)

    # current_labels = np.unique(y)

    valid_labels = set(get_max_label(mode=model_pred_mode))
    # if not set(current_labels).issubset(valid_labels):
    #     raise ValueError(f"Invalid labels found: {set(current_labels) - valid_labels}")
    
    # Convert to numpy arrays
    if isinstance(X, list):
        X = np.array(X)
    if isinstance(y, list):
        y = np.array(y)

    # Create and encode meta information
    # ! change this to fit the abl study
    original_columns = [
        "dataset_aqua", 
        "dataset_commonsenseqa",  
        "dataset_mmlu", 
        "dataset_strategyqa",
        # TODO: add 1B
        'model_Llama-3.2-3B-Instruct',
        "model_Llama3.1-8B-Instruct",  "model_Meta-Llama-3.1-70B-Instruct-Turbo", 
        "method_cot",  "method_l2m",  "method_zero_shot_cot"
    ]

    meta_encoded = create_consistent_meta_encoding(
        config=meta_info,
        X_length=len(X),
        original_columns=original_columns
    )
    # Combine features with meta information
    if model_type == 'lgb':
        X_with_meta = np.hstack([X, meta_encoded]) # (num_chains, 5*50+meta_infos)
    else:
        meta_expanded = np.repeat(meta_encoded[:, np.newaxis, :], X.shape[1], axis=1)
        X_with_meta = np.concatenate((X, meta_expanded), axis=2)

    # Handle label encoding
    if y_scaler is None:
        y_scaler = LabelEncoder()
        y_scaler.fit(valid_labels)

    y_scaled = y_scaler.transform(y)
    
    # Handle feature scaling
    if x_scaler is None:
        raise ValueError("x_scaler is None")
    else:
        if model_type == 'lgb':
            X_scaled = X_with_meta.copy()
            X_scaled = x_scaler.transform(X_with_meta)
        else:
            # original_shape = X_with_meta.shape
            # X_with_meta_flattened = X_with_meta.reshape(original_shape[0], -1)
            # X_scaled_flattened = X_with_meta_flattened.copy()
            # X_scaled_flattened = x_scaler.transform(X_with_meta_flattened)
            # X_scaled = X_scaled_flattened.reshape(original_shape)
            X_scaled = torch.tensor(X_with_meta.astype(np.float32).copy(), device='cpu', dtype=torch.float32)
            X_scaled = rearrange(X_scaled, 'batch seq_len dim -> batch dim seq_len')  # (6980, 15, 30)
            X_scaled = x_scaler(X_scaled)
            X_scaled = rearrange(X_scaled, 'batch dim seq_len -> batch seq_len dim')  # (6980, 30, 15)


    
    # print("==> Evaluating...")
    # Make predictions
    if model_type == 'lgb':
        y_preds = []
        for model in models:
            y_pred = model.predict(X_scaled)
            y_preds.append(y_pred)
        
        raw_y_preds = rearrange(y_preds, 'n_predictors n_samples n_classes -> n_samples n_predictors n_classes')

        # argmax for the predictor's prediction, then voting within the predictors
        y_preds = [mode(np.argmax(predictor_pred, axis=1))[0] for predictor_pred in raw_y_preds]
        
        vote_acc, avg_acc = calculate_accuracies(y_preds, y_scaled, model_pred_mode=model_pred_mode)
    
    else:
        y_preds = []
        model = models
        model.eval()
        correct_preds = 0
        total_preds = 0

        X_test = X_scaled.clone().detach()
        y_test = torch.tensor(y_scaled, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X_test, y_test)

        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        
        raw_y_preds = []
        with torch.no_grad():
            for data, labels in loader:
                outputs = model(data.cuda())
                raw_y_preds.append(outputs[0])
                predicted = (outputs > 0.5).float().cpu().numpy()
                y_preds.append(predicted)
                correct_preds += (predicted == labels.unsqueeze(1)).sum().item()
                total_preds += labels.size(0)

        y_preds = np.concatenate(y_preds, axis=0)
        avg_acc = correct_preds / total_preds
        vote_acc = -1

    # Calculate confidence scores
    results = {
        'voting_acc': vote_acc,
        'avg_acc': avg_acc,
        'predictions': y_preds,
        'raw_predicitons': raw_y_preds,
        'y': y,
    }
    return results


# import copy
# import json
# import os
# import pickle as pkl
# from collections import Counter

# import joblib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import umap.umap_ as umap
# from catboost import CatBoostClassifier
# from einops import rearrange, reduce
# from fire import Fire
# from joblib import dump, load
# from lightgbm import LGBMClassifier
# from scipy.stats import mode
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.manifold import TSNE
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from tqdm import tqdm, trange

# MODEL_NAME_MAPPING = {
#     "Llama-3.2-1B-Instruct": "Llama3_2-1B",
#     "Llama-3.2-3B-Instruct": "Llama3_2-3B",
#     "Meta-Llama-3.1-8B-Instruct-Turbo": "Llama3_1-8B",
#     "Meta-Llama-3.1-70B-Instruct-Turbo": "Llama3_1-70B",
# }

# '''
#     Utils
# '''

# def get_max_label(mode):
#     if mode == 'cls': # for multi-cls classification
#         return ['A', 'B', 'C', 'D', 'E']
#     elif mode == 'reg': # for binary classification
#         return [0, 1]

# def pad_with_threshold(array, threshold=25, MAX_NUM_ANCHORS=5):
#     if array.shape[0] > threshold:
#         # Truncate to threshold length
#         array = array[:threshold, :]
#         pad_length = 0
#         pad_anchor = MAX_NUM_ANCHORS - array.shape[1]
#     else:
#         # Pad to threshold length, and number of anchors
#         pad_length = threshold - array.shape[0]
#         pad_anchor = MAX_NUM_ANCHORS - array.shape[1]

#     padding = ((0, pad_length), (0, pad_anchor))
#     padded_array = np.pad(array, padding, mode='constant', constant_values=0)

#     return padded_array

# def save_models_and_metrics(randomforest_model, 
#                           randomforest_X_scaler,
#                           randomforest_y_scaler, 
#                           llm_train_voting_acc, 
#                           llm_train_avg_acc,
#                           randomforest_mean_std,
#                           ckpt_dir='ckpts'):
    
#     # Create ckpts directory if it doesn't exist
#     os.makedirs(ckpt_dir, exist_ok=True)
    
#     # Save each object
#     dump(randomforest_model, os.path.join(ckpt_dir, 'randomforest_model.pkl'))
#     dump(randomforest_X_scaler, os.path.join(ckpt_dir, 'randomforest_X_scaler.pkl'))
#     dump(randomforest_y_scaler, os.path.join(ckpt_dir, 'randomforest_y_scaler.pkl'))
#     dump(llm_train_voting_acc, os.path.join(ckpt_dir, 'llm_train_voting_acc.pkl'))
#     dump(llm_train_avg_acc, os.path.join(ckpt_dir, 'llm_train_avg_acc.pkl'))
#     dump(randomforest_mean_std, os.path.join(ckpt_dir, 'randomforest_mean_std.pkl'))

# def load_models_and_metrics(ckpt_dir='ckpts'):
#     # Load each object
#     randomforest_model = load(os.path.join(ckpt_dir, 'randomforest_model.pkl'))
#     randomforest_X_scaler = load(os.path.join(ckpt_dir, 'randomforest_X_scaler.pkl'))
#     randomforest_y_scaler = load(os.path.join(ckpt_dir, 'randomforest_y_scaler.pkl'))
#     llm_train_voting_acc = load(os.path.join(ckpt_dir, 'llm_train_voting_acc.pkl'))
#     llm_train_avg_acc = load(os.path.join(ckpt_dir, 'llm_train_avg_acc.pkl'))
#     randomforest_mean_std = load(os.path.join(ckpt_dir, 'randomforest_mean_std.pkl'))
    
#     return (randomforest_model, randomforest_X_scaler, 
#             randomforest_y_scaler, llm_train_voting_acc, 
#             llm_train_avg_acc, randomforest_mean_std)

# '''
#     Dataset Statistics info
# '''
# # LLM original Acc
# def vote_accuracy(list_answers, list_answer_gt_short):
#     voted_answers = [Counter(row).most_common(1)[0][0] if row else '' 
#                     for row in list_answers]
#     return sum(v == g for v, g in zip(voted_answers, list_answer_gt_short)) / len(list_answer_gt_short)

# def row_wise_accuracy(list_answers, list_answer_gt_short):
#     row_accs = [sum(ans == gt for ans in row if ans != '') / len([a for a in row if a != ''])
#                 if any(a != '' for a in row) else 0.0
#                 for row, gt in zip(list_answers, list_answer_gt_short)]
#     return sum(row_accs) / len(row_accs)

# '''
#     Loading Datasets
# '''

# '''
#     File --> Sample data
# '''
# def load_chain_data_and_plot(thoughts_file: str = "None",
#     tool: str = 'tsne',
# ):

#     # load data
#     #######################################
#     assert os.path.exists(thoughts_file), print(thoughts_file)
#     trial_data = json.load(open(thoughts_file, 'r'))
#     model_input = trial_data["model_input"]
#     answers = trial_data["answers"]
#     answer_gt_full = trial_data["answer_gt_full"]
#     answer_gt_short = trial_data["answer_gt_short"]
#     trial_thoughts = trial_data["trial_thoughts"]
#     pkl_path = thoughts_file.replace(".json", f".pkl")
#     pkl_path = pkl_path.replace("thoughts/", "distance_matrix/")  
#     assert os.path.exists(pkl_path)
#     distance_matrix = pkl.load(open(pkl_path, 'rb'))

#     # pre-process for the visualize
#     #######################################
#     chain_color = []
#     labels = []
#     chain_corr = []
#     for [_, answer, binary_label] in trial_thoughts:
#         if binary_label == True:
#             chain_color.append("green")
#             chain_corr.append(binary_label)
#         else:
#             chain_color.append("red")
#             chain_corr.append(binary_label)
#         labels.append(binary_label)

#     # parse thoughts
#     #######################################
#     num_chains = len(trial_thoughts)
#     num_thoughts_each_chain = [len(thoughts) for [thoughts, answer, binary_label] in trial_thoughts]
#     all_answers = [answer for [thoughts, answer, binary_label] in trial_thoughts]
#     all_thoughts = []
#     for [thoughts, answer, binary_label] in trial_thoughts:
#         all_thoughts += thoughts
#     all_thoughts = np.array(all_thoughts)
#     num_all_thoughts = len(all_thoughts)
#     anchors = [model_input] + answers # question and answers
#     num_anchors = len(anchors)
#     anchors_idx_y = [i for i in range(num_anchors)]
#     anchors_idx_x = [(num_all_thoughts + i) for i in range(num_anchors)]

#     # draw the landscape
#     #######################################    
#     if "strategyqa" in thoughts_file:
#         labels_anchors = ["Start", 'A', 'B']
#         if answer_gt_short:
#             answer_gt_short = 'A' # yes
#         else:
#             answer_gt_short = 'B' # no
#         gt_idx = labels_anchors.index(answer_gt_short)
#     else:
#         labels_anchors = ["Start", 'A', 'B', 'C', 'D', 'E']
#         gt_idx = labels_anchors.index(answer_gt_short)
#     answer_idx_y = anchors_idx_y[gt_idx] # the groud truth answer

#     # # normlize thought (T matrix)  
#     processed_distance_matrix = copy.deepcopy(distance_matrix)  
#     # ######################################################
    
#     for chain_idx in range(num_chains): 
#         start_idx, end_idx = sum(num_thoughts_each_chain[:chain_idx]), sum(num_thoughts_each_chain[:chain_idx+1])
#         accumulate_ppl = 0
#         for thought_idx in range(start_idx, end_idx):
#             accumulate_ppl += distance_matrix[thought_idx, 0]
#             # norm D(X, T)
#             processed_distance_matrix[thought_idx, 0] = accumulate_ppl / np.sum(distance_matrix[start_idx:end_idx, 0])
#             # norm D(T, Y)
#             for anchor_idx in range(1, num_anchors):
#                 processed_distance_matrix[thought_idx, anchor_idx] = distance_matrix[thought_idx, anchor_idx] / np.sum(distance_matrix[thought_idx, anchors_idx_y[1:]])

#     # check the normalize effect
#     for chain_idx in range(num_chains): 
#         start_idx, end_idx = sum(num_thoughts_each_chain[:chain_idx]), sum(num_thoughts_each_chain[:chain_idx+1])
#         # assert np.abs(np.sum(processed_distance_matrix[start_idx:end_idx, 0])- 1) < 1e-5 # first col, each chian sum would be 1
#         for thought_idx in range(start_idx, end_idx): # row-sum=1
#             assert ( (np.sum(processed_distance_matrix[start_idx:end_idx, anchors_idx_y[1:]], axis=1) - 1) < 1e-5 ).all()

#     # normlize answer (A matrix)
#     ######################################################
#     A = processed_distance_matrix[num_all_thoughts:]
#     A[np.diag_indices(A.shape[0])] = 0
#     normed_A = copy.deepcopy(A)
#     for col_idx in range(1, num_anchors):
#         normed_A[0][col_idx] = 1 + A[0][col_idx] / np.sum(A[0, anchors_idx_y[1:]])
#     normed_A[:, 0] = normed_A[0, :] # copy same elements to 0-th col
#     normed_A[1:, 1:] = 1 / (num_anchors-1) 
#     normed_A[np.diag_indices(normed_A.shape[0])] = 0
#     processed_distance_matrix[num_all_thoughts:] = normed_A

#     distance_matrix = processed_distance_matrix

#     if tool == 'tsne':
#         tsne = TSNE(n_components=2, perplexity=10, random_state=42)
#         coordinates_2d = tsne.fit_transform(distance_matrix[:, 1:])
#     elif tool == 'umap':
#         coordinates_2d = umap.UMAP(n_neighbors=30, min_dist=0.25, n_components=2, metric='dice', random_state=42).fit_transform(distance_matrix)

#     return distance_matrix, num_chains, num_thoughts_each_chain, coordinates_2d, normed_A, all_answers, answer_gt_short, coordinates_2d[anchors_idx_x[gt_idx], :]

# '''
#     Sample data --> List of Chain data
# '''
# def load_sample_data(
#     model='Meta-Llama-3.1-70B-Instruct-Turbo',
#     dataset='aqua',
#     method="zero_shot_cot",
#     tool='tsne',
#     root='./training_data',
#     start_sample_idx=0,
#     end_sample_idx=10
# ):
#     list_distance_matrix = []
#     list_num_chains = []
#     list_num_thoughts_each_chain = []
#     list_coordinates_2d = []
#     list_normed_A = []
#     list_answers = []
#     list_answer_gt_short = []
#     list_answer_gt_coordinates_2d = []
#     print(f"==> Loading {model}--{method}--{dataset}...")
#     for sample_idx in range(start_sample_idx, end_sample_idx):
#         file_path = f'{root}/{dataset}/thoughts/{model}--{method}--{dataset}--{sample_idx}.json'
#         (
#             distance_matrix, num_chains, num_thoughts_each_chain, coordinates_2d, 
#             normed_A, answers, answer_gt_short, answer_gt_coordinates_2d
#         ) = load_chain_data_and_plot(thoughts_file=file_path, tool=tool)

#         list_distance_matrix.append(distance_matrix)
#         list_num_chains.append(num_chains)
#         list_num_thoughts_each_chain.append(num_thoughts_each_chain)
#         list_coordinates_2d.append(coordinates_2d)
#         list_normed_A.append(normed_A)
#         list_answers.append(answers)
#         list_answer_gt_short.append(answer_gt_short)
#         list_answer_gt_coordinates_2d.append(answer_gt_coordinates_2d)

#     return (
#         list_distance_matrix, list_num_chains, list_num_thoughts_each_chain,
#         list_coordinates_2d, list_answers, list_answer_gt_short, list_normed_A
#     )

# '''
#     List of Chain data --> List of (chain distance matrix, label)
# '''
# def preprocess_data(
#     list_distance_matrix, list_num_chains,
#     list_num_thoughts_each_chain, list_coordinates_2d,
#     list_answers, list_answer_gt_short,
#     mode="cls"
# ):
#     training_data_sample_2d = []
#     training_data_sample_matrix = []

#     for sample_idx in range(len(list_answers)):
#         num_chains = list_num_chains[sample_idx]
#         num_thoughts_each_chain = list_num_thoughts_each_chain[sample_idx]
#         coordinates_2d = list_coordinates_2d[sample_idx]
#         distance_matrix = list_distance_matrix[sample_idx]
#         model_answers = list_answers[sample_idx]
#         gt_answer = list_answer_gt_short[sample_idx]
        
#         training_data_chain_2d = []
#         training_data_chain_matrix = []

#         for chain_idx in range(num_chains):
#             start_idx = sum(num_thoughts_each_chain[:chain_idx])
#             end_idx = sum(num_thoughts_each_chain[:chain_idx+1])
#             chain_coordinates_2d = coordinates_2d[start_idx:end_idx, :]
#             chain_matrix = distance_matrix[start_idx:end_idx, :]

#             if mode == "cls":
#                 training_data_chain_2d.append((chain_coordinates_2d, gt_answer))
#                 training_data_chain_matrix.append((chain_matrix, gt_answer))
#             elif mode == "reg":
#                 chain_answer = model_answers[chain_idx] if model_answers[chain_idx] else "A"
#                 training_data_chain_2d.append((chain_coordinates_2d, gt_answer==chain_answer))
#                 training_data_chain_matrix.append((chain_matrix, gt_answer==chain_answer))
#             else:
#                 raise NotImplementedError

#         training_data_sample_2d.append(training_data_chain_2d)
#         training_data_sample_matrix.append(training_data_chain_matrix)

#     return training_data_sample_2d, training_data_sample_matrix


# def collect_valid_coordinates_and_answers(training_data_sample, dataset):
#     """
#     Process training data sample to collect valid coordinates and answers
    
#     Args:
#         training_data_sample: A list of chains, where each chain contains tuples of (coordinates, answer)
#     Returns:
#         combined_coords: numpy array of coordinates
#         answers_array: numpy array of corresponding answers
#     """
#     all_coords = []
#     all_answers = []

#     try:
#         for chain_idx, chain in enumerate(training_data_sample):
#             # Check if chain is a tuple (should be (coordinates, answer))
#             if isinstance(chain, tuple) and len(chain) == 2:
#                 coordinates, answer = chain
#                 if not isinstance(coordinates, (np.ndarray, list)) or not len(coordinates):
#                     print(f"Empty coordinates at chain {chain_idx}")
#                     continue

#                 try:
#                     coords_array = np.array(coordinates, dtype=np.float64)
#                     if coords_array.size == 0 or np.any(np.isnan(coords_array)):
#                         print(f"Invalid coordinates at chain {chain_idx}")
#                         continue

#                     # remove T(X, Y) if exised
#                     if dataset == "strategyqa":
#                         if coords_array.shape[1] == 3:
#                             coords_array = coords_array[:, 1:] 
#                     elif dataset == "mmlu":
#                         if coords_array.shape[1] == 5:
#                             coords_array = coords_array[:, 1:] 
#                     else:
#                         if coords_array.shape[1] == 6:
#                             coords_array = coords_array[:, 1:] 
    
#                     if coords_array.shape[0] == 1: continue # remove the chain with only one thought

#                     # this ensure the coords_array as (N, num_anchors)
#                     coords_array = pad_with_threshold(coords_array, threshold=30) # ! truncate/pad the number of thought
#                     all_coords.append(coords_array) # ! make sure all the coordinates are in the same shape (N, D)
#                     all_answers.append(answer)

#                 except (ValueError, TypeError) as e:
#                     print(f"Error processing coordinates at chain {chain_idx}: {e}")
#                     continue
#             else:
#                 print(f"Invalid chain format at index {chain_idx}")
#                 continue

#     except Exception as e:
#         print(f"Error in data processing: {e}")
#         return None, None

#     if not all_coords:
#         print("No valid coordinates collected")
#         return None, None

#     try:
#         # TODO: use RNN/LSTM, so we can directly use (n, l, d) as input
#         combined_coords = np.array(all_coords) 
#         # # NOTE: this will be (N, 30*5), N is the number of samples
#         combined_coords = rearrange(combined_coords, 'n l d -> n (l d)') 
#         # # using max/mean/sum pooling
#         # X_mean = reduce(combined_coords, 'n l d -> n d', 'mean')  # Reduce length dimension with mean
#         # X_max = reduce(combined_coords, 'n l d -> n d', 'max')    # Reduce length dim/ension with max
#         # X_sum = reduce(combined_coords, 'n l d -> n d', 'sum')    # Reduce length dimension with sum
#         # # Combine features
#         # # # NOTE: this will be (N, 3*5), N is the number of samples
#         # combined_coords = np.hstack([X_mean, X_max, X_sum]) # n, 3*d

#         answers_array = np.array(all_answers)
#         return combined_coords, answers_array

#     except Exception as e:
#         for coord in all_coords:
#             print(f"Coordinates: {coord.shape}")
#         print(f"Error during final processing: {e}")
#         return None, None

# '''
#     Multiple config datasets --> X, y, meta_info
# '''
# def prepare_data_for_training(dataset_configs, verbose=False, mode='reg'):
#     """
#     Prepare data for training including original accuracy calculations
#     """
#     all_data = []
#     meta_info = []
#     label_sets = set()
#     voting_accs = []
#     average_accs = []
    
#     # Loading Raw data
#     ##########################################
#     for config in dataset_configs:
#         list_distance_matrix, list_num_chains, list_num_thoughts_each_chain, \
#         list_coordinates_2d, list_answers, list_answer_gt_short, list_normed_A = \
#             load_sample_data(
#                 model=config['model'],
#                 dataset=config['dataset'],
#                 method=config['method'],
#                 start_sample_idx=config['start_idx'],
#                 end_sample_idx=config['end_idx']
#             )
        
#         # Calculate original accuracies
#         voting_acc = vote_accuracy(list_answers, list_answer_gt_short)
#         average_acc = row_wise_accuracy(list_answers, list_answer_gt_short)
        
#         voting_accs.append(voting_acc)
#         average_accs.append(average_acc)
        
#         # Process data
#         _, training_data_matrix = preprocess_data(
#             list_distance_matrix, list_num_chains,
#             list_num_thoughts_each_chain, list_coordinates_2d,
#             list_answers, list_answer_gt_short,
#             mode=mode,
#         )

#         # NOTE: we use the full distance matrix for training
#         for sample_data in training_data_matrix:
#             X, y = collect_valid_coordinates_and_answers(sample_data, dataset=config['dataset'])
#             if X is not None and y is not None:
#                 current_labels = np.unique(y)

                
#                 if not set(current_labels).issubset(set([0, 1,])): 
#                     if verbose:
#                         print(f"Warning: Dataset {config['dataset']} contains invalid labels: {set(current_labels) - set(['A', 'B', 'C', 'D', 'E'])}")
#                     continue

#                 label_sets.update(current_labels)
#                 all_data.append((X, y)) # chains X: (10, D); y: (10, )
#                 meta_info.extend([{
#                     'dataset': config['dataset'],
#                     'model': config['model'],
#                     'method': config['method'],
#                 }] * len(X))

#     ##########################################
#     # Combine all data
#     # First, check dimensions
#     X_list = []
#     y_list = []

#     for X, y in all_data:
#         if isinstance(X, list):
#             X = np.array(X)
#         if isinstance(y, list):
#             y = np.array(y)
#         X_list.append(X)
#         y_list.append(y)

#     # Concatenate along the first dimension (number of anchors)
#     X_combined = np.concatenate(X_list, axis=0)
#     y_combined = np.concatenate(y_list, axis=0)

#     # Convert meta information
#     meta_df = pd.DataFrame(meta_info)
#     meta_encoded = pd.get_dummies(meta_df, columns=['dataset', 'model', 'method'])
#     # Combine features with meta information
#     X_with_meta = np.hstack([X_combined, meta_encoded.values]) # (num_chains, 5*50+meta_infos)

#     # Prepare for training
#     label_encoder = LabelEncoder()
#     label_encoder.fit(get_max_label(mode=mode))
    
#     scaler = StandardScaler()
#     X_scaled = X_with_meta.copy()
#     X_scaled = scaler.fit_transform(X_with_meta)

#     y_scaled = label_encoder.transform(y_combined)
    
#     if verbose:
#         print("\nData Processing Summary:")
#         print(f"Total samples: {len(y_combined)}")
#         print(f"Feature dimensionality: {X_with_meta.shape[1]}")
#         print(f"Number of unique labels: {len(label_sets)}")
#         print(f"Labels present: {sorted(list(label_sets))}")
#         print(f"\nAverage LLM Performance:")
#         print(f"Average Voting Accuracy: {np.mean(voting_accs):.2f}")
#         print(f"Average Chain Accuracy: {np.mean(average_accs):.2f}")
    

#     acc_infos = {
#         'voting_accs': voting_accs,
#         'average_accs': average_accs,
#         'mean_voting_acc': np.mean(voting_accs),
#         'mean_average_acc': np.mean(average_accs)
#     }

#     return (X_scaled, y_scaled, scaler, label_encoder, acc_infos)

# '''
#     Evaluation
# '''

# '''
#     List of Chain data --> (chain distance matrix, label)

#     NOTE: This is similar to the `preprocess_data`, 
#     but loading single sample data, instead of all samples.
# '''

# def load_chain_data(
#         sample_idx,
#         list_num_chains,
#         list_num_thoughts_each_chain,
#         list_coordinates_2d,
#         list_distance_matrix,
#         list_answers,
#         list_answer_gt_short,
#         mode="cls"
#     ):
#     num_chains = list_num_chains[sample_idx]
#     num_thoughts_each_chain = list_num_thoughts_each_chain[sample_idx]
#     coordinates_2d = list_coordinates_2d[sample_idx]
#     distance_matrix = list_distance_matrix[sample_idx]
#     model_answers = list_answers[sample_idx]
#     gt_answer = list_answer_gt_short[sample_idx]

#     training_data_chain_2d = []
#     training_data_chain_matrix = []
#     for chain_idx in range(num_chains):
#         start_idx, end_idx = sum(num_thoughts_each_chain[:chain_idx]), sum(num_thoughts_each_chain[:chain_idx+1])
#         chain_coordinates_2d = coordinates_2d[start_idx:end_idx, :]
#         chain_matrix = distance_matrix[start_idx:end_idx, :]
#         if mode == "cls":
#             training_data_chain_2d.append((chain_coordinates_2d, gt_answer)) # for classification
#             training_data_chain_matrix.append((chain_matrix, gt_answer))
#         elif mode == "reg":
#             chain_answer = model_answers[chain_idx] if model_answers[chain_idx] else "A" # default answer for the chain with empty values
#             training_data_chain_2d.append((chain_coordinates_2d, gt_answer==chain_answer)) # for regression
#             training_data_chain_matrix.append((chain_matrix, gt_answer==chain_answer)) # for regression
#         else:
#             raise NotImplementedError
        
#     return training_data_chain_2d, training_data_chain_matrix

# def chain_collect_valid_coords_and_answers(list_chain_coord_answers, dataset):
#     all_coords = []
#     all_answers = []
#     for chain_idx, (coordinates, answer) in enumerate(list_chain_coord_answers):
#         if not len(coordinates):
#             print(f"Empty coordinates at chain {chain_idx}")
#             continue
            
#         try:
#             coords_array = np.array(coordinates, dtype=np.float64)
            
#             if coords_array.size == 0 or np.any(np.isnan(coords_array)):
#                 print(f"Invalid coordinates at chain {chain_idx}")
#                 continue
                
#             # remove T(X, Y) if exised
#             if dataset == "strategyqa":
#                 if coords_array.shape[1] == 3:
#                     coords_array = coords_array[:, 1:] 
#             elif dataset == "mmlu":
#                 if coords_array.shape[1] == 5:
#                     coords_array = coords_array[:, 1:] 
#             else:
#                 if coords_array.shape[1] == 6:
#                     coords_array = coords_array[:, 1:] 
            
#             # truncate/pad the number of thought, with maximun of threshold (default: 50)
#             # this ensure the coords_array as (N, 5)
#             coords_array = pad_with_threshold(coords_array, threshold=30) 
            
#             # make sure all the coordinates are in the same shape (N, D)
#             all_coords.append(coords_array) 
#             all_answers.append(answer)

#         except (ValueError, TypeError) as e:
#             print(f"Error processing coordinates at chain {chain_idx}: {e}")
#             continue

#     if not all_coords:
#         print("No valid coordinates collected")
#         return None, None

#     try:
#         combined_coords = np.array(all_coords) 
#         # # NOTE: this will be (N, 30*5), N is the number of samples
#         combined_coords = rearrange(combined_coords, 'n l d -> n (l d)') 
#         # using max/mean/sum pooling
#         # X_mean = reduce(combined_coords, 'n l d -> n d', 'mean')  # Reduce length dimension with mean
#         # X_max = reduce(combined_coords, 'n l d -> n d', 'max')    # Reduce length dim/ension with max
#         # X_sum = reduce(combined_coords, 'n l d -> n d', 'sum')    # Reduce length dimension with sum
#         # Combine features
#         # # NOTE: this will be (N, 3*5), N is the number of samples
#         # combined_coords = np.hstack([X_mean, X_max, X_sum]) # n, 3*d

#         answers_array = np.array(all_answers)

#         # if mode == "cls":
#         #     answers_array = np.unique(np.array(all_answers))
#         #     assert len(answers_array) == 1, "Answers are not consistent within the chain"

#         return combined_coords, answers_array

#     except Exception as e:
#         print(f"Error during final processing: {e}")
#         return None, None

# def calculate_accuracies(y_preds, y_true, model_pred_mode='reg'):
#     # Voting
#     if model_pred_mode == 'cls':
#         vote_acc = 1 if mode(y_preds)[0] == y_true else 0
#     else:
#         vote_acc = -1
#     # average accuracy
#     correct_predictions = np.sum(y_preds == y_true)
#     avg_acc = correct_predictions / len(y_preds)

#     return vote_acc, avg_acc

# def create_consistent_meta_encoding(config, X_length, original_columns):
#     """
#     创建与训练数据一致的meta编码
    
#     Args:
#         config: 包含 dataset, model, method 的配置
#         X_length: 样本数量
#         original_columns: 原始训练数据的编码列名列表
#     """
#     # 创建单个样本的meta信息
#     meta_info = [{
#         'dataset': config['dataset'],
#         'model': config['model'],
#         'method': config['method'],
#     }] * X_length
    
#     # 转换为DataFrame
#     meta_df = pd.DataFrame(meta_info)
    
#     # 使用相同的列进行one-hot编码
#     meta_encoded = pd.get_dummies(meta_df, columns=['dataset', 'model', 'method'])
    
#     # 确保所有原始列都存在，没有的填0
#     for col in original_columns:
#         if col not in meta_encoded.columns:
#             meta_encoded[col] = 0
            
#     # 确保列的顺序一致
#     meta_encoded = meta_encoded[original_columns]
    
#     return meta_encoded.values

# '''
#     Random Forest Evaluation
# '''

# def eval_random_forest(
#         data: list = None, 
#         meta_info: dict = None,
#         x_scaler: StandardScaler = None,
#         y_scaler: StandardScaler = None,
#         models: list = [LGBMClassifier],
#         model_pred_mode: str = 'cls'
#     ):
#     # X: (num_chains, 50*5+meta_info_dim)
#     # y: (num_chains, )
#     X, y = chain_collect_valid_coords_and_answers(data, dataset=meta_info['dataset'])

#     # current_labels = np.unique(y)

#     valid_labels = set(get_max_label(mode=model_pred_mode))
#     # if not set(current_labels).issubset(valid_labels):
#     #     raise ValueError(f"Invalid labels found: {set(current_labels) - valid_labels}")
    
#     # Convert to numpy arrays
#     if isinstance(X, list):
#         X = np.array(X)
#     if isinstance(y, list):
#         y = np.array(y)

#     # Create and encode meta information
#     original_columns = [
#         "dataset_aqua", "dataset_commonsenseqa",  "dataset_mmlu", "dataset_strategyqa",
#         # TODO: add 1B and 3B
#         "model_Llama3.1-8B-Instruct",  "model_Meta-Llama-3.1-70B-Instruct-Turbo", 
#         "method_cot",  "method_l2m",  "method_zero_shot_cot"
#     ]

#     meta_encoded = create_consistent_meta_encoding(
#         config=meta_info,
#         X_length=len(X),
#         original_columns=original_columns
#     )
#     # Combine features with meta information
#     X_with_meta = np.hstack([X, meta_encoded])

#     # Handle label encoding
#     if y_scaler is None:
#         y_scaler = LabelEncoder()
#         y_scaler.fit(valid_labels)

#     y_scaled = y_scaler.transform(y)

#     # Handle feature scaling
#     if x_scaler is None:
#         raise ValueError("x_scaler is None")
#     else:
#         X_scaled = X_with_meta.copy()
#         X_scaled = x_scaler.transform(X_with_meta)

#     # print("==> Evaluating...")
#     # Make predictions
#     y_preds = []
#     for model in models:
#         y_pred = model.predict(X_scaled)
#         y_preds.append(y_pred)
    
#     y_preds = rearrange(y_preds, 'n_predictors n_samples n_classes -> n_samples n_predictors n_classes')

#     # argmax for the predictor's prediction, then voting within the predictors
#     y_preds = [mode(np.argmax(predictor_pred, axis=1))[0] for predictor_pred in y_preds]
    
#     vote_acc, avg_acc = calculate_accuracies(y_preds, y_scaled, model_pred_mode=model_pred_mode)

#     # Calculate confidence scores
#     results = {
#         'voting_acc': vote_acc,
#         'avg_acc': avg_acc,
#         'predictions': y_preds,
#         'y': y,
#     }
#     return results


