"""
Landscape visualization utilities for LOT.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from .utils import process_chain_points, split_list


def draw_landscape(
    dataset_name: str, 
    plot_datas: Dict[int, Dict[str, Any]], 
    splited_T_2D: List[np.ndarray], 
    A_matrix_2D: np.ndarray, 
    num_all_thoughts_w_start_list: List[int]
) -> go.Figure:
    """
    Draw a landscape visualization of reasoning traces.
    
    Args:
        dataset_name (str): Name of the dataset.
        plot_datas (Dict[int, Dict[str, Any]]): Data for plotting.
        splited_T_2D (List[np.ndarray]): Split T matrix in 2D.
        A_matrix_2D (np.ndarray): A matrix in 2D.
        num_all_thoughts_w_start_list (List[int]): List of number of thoughts with start.
        
    Returns:
        go.Figure: Plotly figure object.
    """
    all_T_with_start_coordinate_matrix = split_list(num_all_thoughts_w_start_list, splited_T_2D)

    column_titles = [r'0-20% states', r'20-40% states', r'40-60% states', r'60-80% states', r'80-100% states']
    fig = make_subplots(rows=2, cols=5, 
                        vertical_spacing=0.01, horizontal_spacing=0.005,
                        column_titles=column_titles)

    # Collect points and separate them for correct/wrong chains
    wrong_chain_points = []
    correct_chain_points = []
    all_start_coordinates = []
    for sample_idx, plot_data in plot_datas.items():

        num_thoughts_each_chain, num_chains, _, all_answers, answer_gt_short = plot_data.values()
        try:
            temp_distance_matrix = all_T_with_start_coordinate_matrix[sample_idx]
        except:
            print(len(all_T_with_start_coordinate_matrix), sample_idx)
        thoughts_coordinates = np.array(temp_distance_matrix[:-1])
        start_coordinate = temp_distance_matrix[-1]
        all_start_coordinates.append(start_coordinate)
        
        # Collect points for each chain
        for chain_idx in range(num_chains):
            start_idx = sum(num_thoughts_each_chain[:chain_idx])
            end_idx = sum(num_thoughts_each_chain[:chain_idx+1])
            
            if end_idx <= start_idx:
                continue

            chain_points = thoughts_coordinates[start_idx:end_idx]

            if len(chain_points) <= 1:
                continue

            chain_data = {
                'points': chain_points,
                'start': start_coordinate
            }

            if all_answers[chain_idx] == answer_gt_short:
                correct_chain_points.append(chain_data)
            else:
                wrong_chain_points.append(chain_data)

    # Process both chains first
    wrong_x, wrong_y, wrong_weights, _, _ = process_chain_points(wrong_chain_points)
    correct_x, correct_y, correct_weights, _, _ = process_chain_points(correct_chain_points)

    # Calculate thresholds for both sets
    wrong_thresholds = np.percentile(wrong_weights, [20, 40, 60, 80]) if len(wrong_weights) > 0 else np.array([0.2, 0.4, 0.6, 0.8])
    
    # Handle the case where there are no correct answers
    if len(correct_weights) > 0:
        correct_thresholds = np.percentile(correct_weights, [20, 40, 60, 80])
    else:
        print("Warning: No correct answers found. Using default thresholds for correct answers.")
        correct_thresholds = np.array([0.2, 0.4, 0.6, 0.8])
        # Create empty arrays for correct segments
        correct_x = np.array([])
        correct_y = np.array([])

    # Lists to store segment data
    wrong_segments = []
    correct_segments = []
    # Process wrong chains
    for i in range(5):
        if i == 0:
            wrong_mask = wrong_weights <= wrong_thresholds[0] if len(wrong_weights) > 0 else np.array([], dtype=bool)
            correct_mask = correct_weights <= correct_thresholds[0] if len(correct_weights) > 0 else np.array([], dtype=bool)
        elif i == 4:
            wrong_mask = wrong_weights > wrong_thresholds[3] if len(wrong_weights) > 0 else np.array([], dtype=bool)
            correct_mask = correct_weights > correct_thresholds[3] if len(correct_weights) > 0 else np.array([], dtype=bool)
        else:
            if len(wrong_weights) > 0:
                wrong_mask = (wrong_weights > wrong_thresholds[i-1]) & (wrong_weights <= wrong_thresholds[i])
            else:
                wrong_mask = np.array([], dtype=bool)
                
            if len(correct_weights) > 0:
                correct_mask = (correct_weights > correct_thresholds[i-1]) & (correct_weights <= correct_thresholds[i])
            else:
                correct_mask = np.array([], dtype=bool)

        # Get segments for both wrong and correct
        if len(wrong_weights) > 0:
            wrong_x_segment = np.array(wrong_x)[wrong_mask]
            wrong_y_segment = np.array(wrong_y)[wrong_mask]
        else:
            wrong_x_segment = np.array([])
            wrong_y_segment = np.array([])
            
        if len(correct_weights) > 0:
            correct_x_segment = np.array(correct_x)[correct_mask]
            correct_y_segment = np.array(correct_y)[correct_mask]
        else:
            correct_x_segment = np.array([])
            correct_y_segment = np.array([])

        # Store segments and their scales
        wrong_segments.append((wrong_x_segment, wrong_y_segment))
        correct_segments.append((correct_x_segment, correct_y_segment))

    # Plot wrong chains (top subplot)
    #######################################
    for i in range(5):
        wrong_x_segment, wrong_y_segment = wrong_segments[i]
        correct_x_segment, correct_y_segment = correct_segments[i]

        # Only add trace if there are points in the segment
        if len(wrong_x_segment) > 0:
            fig.add_trace(
                go.Histogram2dContour(
                    x=wrong_x_segment,
                    y=wrong_y_segment,
                    colorscale="Reds",
                    showscale=False,
                    histfunc='count',
                    contours=dict(
                        showlines=True,
                        coloring='fill'
                    ),
                    autocontour=True,
                    opacity=0.6,
                    name=f'Wrong Range {i+1}'
                ),
                row=1, col=i+1
            )
        else:
            # Add an empty trace to maintain the subplot structure
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode='markers',
                    showlegend=False
                ),
                row=1, col=i+1
            )

        # Only add trace if there are points in the segment
        if len(correct_x_segment) > 0:
            fig.add_trace(
                go.Histogram2dContour(
                    x=correct_x_segment,
                    y=correct_y_segment,
                    colorscale="Blues",
                    showscale=False,
                    histfunc='count',
                    contours=dict(
                        showlines=True,
                        coloring='fill'
                    ),
                    autocontour=True,
                    opacity=0.6,
                    name=f'Correct Range {i+1}'
                ),
                row=2, col=i+1
            )
        else:
            # Add an empty trace to maintain the subplot structure
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode='markers',
                    showlegend=False
                ),
                row=2, col=i+1
            )

    # Add anchors to both plots
    if dataset_name == "mmlu":
        labels_anchors = ['A', 'B', 'C', 'D']
    elif dataset_name == "strategyqa":
        labels_anchors = ['A', 'B']
    else:
        labels_anchors = ['A', 'B', 'C', 'D', 'E']

    # Add anchors to both subplots
    for idx, anchor_name in enumerate(labels_anchors):
        if idx == 0:  # the first anchor is the correct one 
            marker_symbol = 'star'
            marker_color = "green"
        else: 
            marker_symbol = 'x'
            marker_color = "red"

        # Add to top subplot
        for col_idx in range(5):
            fig.add_trace(
                go.Scatter(
                    x=[A_matrix_2D[idx, 0]], 
                    y=[A_matrix_2D[idx, 1]], 
                    mode='markers',
                    marker=dict(
                        symbol=marker_symbol, 
                        size=18, 
                        line_width=0.5, 
                        color=marker_color,
                        opacity=0.8, # transparency
                    ),
                    showlegend=False,
                ),
                row=1, col=col_idx+1
            )

        # Add to bottom subplot
        for col_idx in range(5):
            fig.add_trace(
                go.Scatter(
                    x=[A_matrix_2D[idx, 0]], 
                    y=[A_matrix_2D[idx, 1]], 
                    mode='markers',
                    marker=dict(
                        symbol=marker_symbol, 
                        size=18, 
                        line_width=0.5, 
                        color=marker_color,
                        opacity=0.8, # transparency
                    ),
                    showlegend=False,
                ),
                row=2, col=col_idx+1
            )

    # Move the subplot title to bottom
    fig = move_titles_to_bottom(fig, column_titles=column_titles, y_position=-0.12)

    # Update both subplots to remove axes and maintain same range
    for row in [1, 2]:
        for i in range(1, 6):
            fig.update_xaxes(
                row=row, 
                col=i,
                showticklabels=False,
                showgrid=True,           # Enable grid
                gridwidth=1,             # Grid line width
                gridcolor='lightgray',   # Grid line color
                zeroline=False,          # Hide zero line
                showline=False,          # Show axis line
                linewidth=1,             # Axis line width
                linecolor='black',       # Axis line color
                mirror=True,             # Mirror axis line
            )
            fig.update_yaxes(
                row=row, 
                col=i,
                showticklabels=False,
                showgrid=True,           # Enable grid
                gridwidth=1,             # Grid line width
                gridcolor='lightgray',   # Grid line color
                zeroline=False,          # Hide zero line
                showline=False,          # Show axis line
                linewidth=1,             # Axis line width
                linecolor='black',       # Axis line color
                mirror=True,             # Mirror axis line
            )

    # Update layout for a tighter plot
    fig.update_layout(
        height=350,  # Reduced height
        width=1500,
        margin=dict(l=5, r=5, t=5, b=37),  # Remove margins
        plot_bgcolor='white',  # Set plot background to white
        paper_bgcolor='white', # Set paper background to white
    )
    template = dict(
        layout=dict(
            font_color="black",
            paper_bgcolor="white",
            plot_bgcolor="white",
            title_font_color="black",
            legend_font_color="black",
            
            xaxis=dict(
                title_font_color="black",
                tickfont_color="black",
                linecolor="black",
                gridcolor="black",
                zerolinecolor="black",
            ),
            
            yaxis=dict(
                title_font_color="black", 
                tickfont_color="black",
                linecolor="black",
                gridcolor="black",
                zerolinecolor="black",
            ),
            
            hoverlabel=dict(
                font_color="black",
                bgcolor="white"
            ),
            
            annotations=[dict(font_color="black")],
            shapes=[dict(line_color="black")],
            
            coloraxis=dict(
                colorbar_tickfont_color="black",
                colorbar_title_font_color="black"
            ),
        )
    )

    fig.update_layout(template=template)

    return fig


def move_titles_to_bottom(fig, column_titles, y_position=-0.12):
    """
    Move column titles to the bottom of the figure.
    
    Args:
        fig (go.Figure): Plotly figure object.
        column_titles (List[str]): List of column titles.
        y_position (float): Y position for the titles.
        
    Returns:
        go.Figure: Updated figure.
    """
    for i, title in enumerate(column_titles):
        fig.add_annotation(
            x=fig.layout.annotations[i].x,
            y=y_position,
            text=title,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=22),
            xanchor="center"  # Center the text horizontally
        )
    
    # Remove the original titles
    fig.update_layout(annotations=fig.layout.annotations)
    
    return fig

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

def process_landscape_data(
    model: str,
    dataset: str,
    models: List[str] = ["Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Meta-Llama-3.1-8B-Instruct-Turbo", "Meta-Llama-3.1-70B-Instruct-Turbo"],
    methods: List[str] = ["cot", "l2m", "mcts", "tot"],
    plot_type: str = 'method',
    ROOT: str = "./exp-data"
) -> Tuple[List[np.ndarray], np.ndarray, List[Dict[int, Dict[str, Any]]], List[List[int]]]:
    """
    Process data for landscape visualization.
    
    Args:
        model (str): Model name.
        dataset (str): Dataset name.
        methods (List[str]): List of methods to process.
        plot_type (str): Type of plot ('method' or 'model').
        ROOT (str): Root directory for data.
        
    Returns:
        Tuple containing:
            - list_all_T_2D: List of T matrices in 2D.
            - A_matrix_2D: A matrix in 2D.
            - list_plot_data: List of plot data.
            - list_num_all_thoughts_w_start_list: List of number of thoughts with start.
    """
    from sklearn.manifold import TSNE
    
    distance_matrix_shape = []
    list_distance_matrix = []
    list_num_all_thoughts_w_start_list = []
    list_plot_data = []

    # Aggregate and t-SNE down-project the data from different models
    if plot_type == "model":
        for model_name in models:
            distance_matries, num_all_thoughts_w_start_list, plot_datas = load_landscape_data(model=model_name, dataset=dataset, method=methods[0], ROOT=ROOT)
            list_distance_matrix.append(distance_matries)
            list_plot_data.append(plot_datas)
            list_num_all_thoughts_w_start_list.append(num_all_thoughts_w_start_list)
            distance_matrix_shape.append(distance_matries.shape)
    
    elif plot_type == "dataset":
        # We cannot make all the samples with different num_answer to process together
        raise NotImplementedError
    
    # Aggregate and t-SNE down-project the data from different methods
    elif plot_type == "method":
        for method in methods:
            distance_matries, num_all_thoughts_w_start_list, plot_datas = load_landscape_data(model=model, dataset=dataset, method=method, ROOT=ROOT)
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

    # Concatenate all T and A(0-th row) (Nx(num_thoughts + 1), C), then concat the constant A matrix (C, C)
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    all_T_constant_A_distance_matrix = tsne.fit_transform(np.concatenate([fig_data, target_A_matrix]))

    # Split the Nx(num_thoughts + 1) back to sample-wise distance matrix
    if dataset == "mmlu":
        index = -4
    elif dataset == "strategyqa":
        index = -2
    else:
        index = -5
    all_T_2D, A_matrix_2D = all_T_constant_A_distance_matrix[:index, :], all_T_constant_A_distance_matrix[index:, :]
    list_all_T_2D = split_array(distance_matrix_shape, all_T_2D)

    return list_all_T_2D, A_matrix_2D, list_plot_data, list_num_all_thoughts_w_start_list


def load_landscape_data(
    model: str,
    dataset: str,
    method: str = "cot",
    ROOT: str = "./exp-data"
) -> Tuple[np.ndarray, List[int], Dict[int, Dict[str, Any]]]:
    """
    Load data for landscape visualization.
    
    Args:
        model (str): Model name.
        dataset (str): Dataset name.
        method (str): Method name.
        ROOT (str): Root directory for data.
        
    Returns:
        Tuple containing:
            - distance_matrices: Concatenated distance matrices.
            - num_all_thoughts_w_start_list: List of number of thoughts with start.
            - plot_datas: Dictionary of plot data.
    """
    import json
    import pickle as pkl
    
    # Load data
    plot_datas = {} 
    distance_matrices = []
    num_all_thoughts_w_start_list = []
    
    # Get the list of files in the distance_matrix directory
    distance_matrix_dir = f'{ROOT}/{dataset}/distance_matrix/'
    if not os.path.exists(distance_matrix_dir):
        raise FileNotFoundError(f"Directory not found: {distance_matrix_dir}")
    
    # Filter files for the specific model and method
    files = [f for f in os.listdir(distance_matrix_dir) if f.startswith(f"{model}--{method}--{dataset}--") and f.endswith(".pkl")]
    
    # Sort files by index
    files.sort(key=lambda x: int(x.split("--")[-1].split(".")[0]))
    
    for file_name in tqdm(files, desc=f"Loading plot data for {model} {method} {dataset}"):
        sample_idx = int(file_name.split("--")[-1].split(".")[0])
        
        # Load thoughts file
        thoughts_file = f'{ROOT}/{dataset}/thoughts/{model}--{method}--{dataset}--{sample_idx}.json'

        if not os.path.exists(thoughts_file):
            print(f"Thoughts file not found: {thoughts_file}")
            continue
        
        # Load distance matrix
        distance_matrix_file = f'{ROOT}/{dataset}/distance_matrix/{file_name}'
        if not os.path.exists(distance_matrix_file):
            print(f"Distance matrix file not found: {distance_matrix_file}")
            continue
        
        # Load the trial data from the JSON file
        with open(thoughts_file, 'r', encoding='utf-8') as f:
            trial_data = json.load(f)
        
        # Extract the required fields from the trial data
        trial_thoughts = trial_data["trial_thoughts"]
        all_answers = [answer for _, answer, _ in trial_thoughts]
        answer_gt_short = trial_data["answer_gt_short"]
        
        # Calculate num_thoughts_each_chain
        num_thoughts_each_chain = [len(thoughts) for thoughts, _, _ in trial_thoughts]
        num_chains = len(trial_thoughts)
        num_all_thoughts = sum(num_thoughts_each_chain)
        
        # Load distance matrix
        with open(distance_matrix_file, 'rb') as f:
            distance_matrix = pkl.load(f)
        
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
        
        # Skip the broken distance matrix
        expected_dims = {
            "commonsenseqa": 6,
            "aqua": 6,
            "mmlu": 5,
            "strategyqa": 3,
            "dummy": 6
        }
        if distance_matrix.shape[1] != expected_dims.get(dataset):
            continue

        # Normalize the distance matrix
        distance_matrix = distance_matrix[:num_all_thoughts+1, 1:] # get T matrix and the first row of the A matrix
        distance_matrix = distance_matrix / np.linalg.norm(distance_matrix, axis=1, ord=1, keepdims=True) # normalize the D (T, Y)
        
        # sort the source_matrix to make the GT at the first row (GT, Y_c, ... other answers)
        distance_matrix = rearrange_columns(distance_matrix, gt_idx-1)

        # Store data
        plot_datas[sample_idx] = {
            "num_thoughts_each_chain": num_thoughts_each_chain,
            "num_chains": num_chains,
            "num_all_thoughts": num_all_thoughts,
            "all_answers": all_answers,
            "answer_gt_short": answer_gt_short
        }
        
        distance_matrices.append(distance_matrix)
        num_all_thoughts_w_start_list.append(num_all_thoughts+1)  # add one row from A matrix
    
    if len(distance_matrices) == 0:
        raise ValueError(f"No data found for {model} {method} {dataset}")
    
    # Concatenate all distance matrices
    distance_matrices = np.concatenate(distance_matrices)
    
    return distance_matrices, num_all_thoughts_w_start_list, plot_datas


def split_array(shapes, array):
    """
    Split an array according to the given shapes.
    
    Args:
        shapes (List[Tuple[int, int]]): List of shapes.
        array (np.ndarray): Array to split.
        
    Returns:
        List[np.ndarray]: List of split arrays.
    """
    result = []
    start_idx = 0
    for shape in shapes:
        end_idx = start_idx + shape[0]
        result.append(array[start_idx:end_idx])
        start_idx = end_idx
    return result 