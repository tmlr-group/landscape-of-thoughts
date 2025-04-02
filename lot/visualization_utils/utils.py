import copy
import json
import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# import umap.umap_ as umap
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
    print(thoughts_file)
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

def loading_data_from_file(model='Meta-Llama-3.1-70B-Instruct-Turbo', dataset='aqua', method="cot", ROOT="./Landscape-Data"):
    # Load data
    ########################################
    plot_datas = {} 
    distance_matries = []
    num_all_thoughts_w_start_list = []
    total_sample = len(os.listdir(f'{ROOT}/{dataset}/distance_matrix/'))
    for sample_idx in tqdm(range(total_sample), ncols=total_sample):
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

def process_data(
        model='Meta-Llama-3.1-70B-Instruct-Turbo', 
        dataset='aqua', 
        methods=["cot"], 
        plot_type='method', 
        ROOT="./Landscape-Data", 
    ):
    distance_matrix_shape = []
    list_distance_matrix = []
    list_num_all_thoughts_w_start_list = []
    list_plot_data = []

    if plot_type == "model":
        # assert method == 'cot', "model should be cot"
        # assert dataset == 'aqua', "dataset should be aqua"
        for model in ['Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct', 'Meta-Llama-3.1-8B-Instruct-Turbo', 'Meta-Llama-3.1-70B-Instruct-Turbo']:
            distance_matries, num_all_thoughts_w_start_list, plot_datas = loading_data_from_file(model=model, dataset=dataset, methods=methods, ROOT=ROOT)
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
        for method in methods:
            distance_matries, num_all_thoughts_w_start_list, plot_datas = loading_data_from_file(model=model, dataset=dataset, method=method, ROOT=ROOT)
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


'''
    Single Chain Plot
'''


def process_single_thought_file(thoughts_file: str = "None",
         tool: str = 'tsne',
         remove_denominator: bool = False,
):

    # load data
    #######################################
    assert os.path.exists(thoughts_file) 
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
        labels_anchors = ["Start", 'yes', 'no']
        if answer_gt_short:
            answer_gt_short = 'yes'
        else:
            answer_gt_short = 'no'
        gt_idx = labels_anchors.index(answer_gt_short)
    elif "mmlu" in thoughts_file:
        labels_anchors = ["Start", 'A', 'B', 'C', 'D']
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
            for anchor_idx in range(1, num_anchors):
                processed_distance_matrix[thought_idx, anchor_idx] = distance_matrix[thought_idx, anchor_idx] / np.sum(distance_matrix[thought_idx, anchors_idx_y[1:]])

    # check the normalize effect
    for chain_idx in range(num_chains): 
        start_idx, end_idx = sum(num_thoughts_each_chain[:chain_idx]), sum(num_thoughts_each_chain[:chain_idx+1])
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

    coordinates_value = distance_matrix[:, answer_idx_y] / np.sum(distance_matrix[:, anchors_idx_y[1:]], axis=1) # v2 normalize
    if tool == 'tsne':
        # NOTE: The perplexity must be less than the number of samples
        # perplexity is a sensitive in tuning
        # treat the distance matrix as embeddings
        tsne = TSNE(n_components=2, perplexity=10, random_state=42)
        # ori
        # only use D(T, Y) for drawing 
        coordinates_2d = tsne.fit_transform(distance_matrix[:, 1:])
    elif tool == 'umap':
        # treat the distance matrix as embeddings
        coordinates_2d = umap.UMAP(n_neighbors=30, min_dist=0.25, n_components=2, metric='dice', random_state=42).fit_transform(distance_matrix)


    return coordinates_2d, num_thoughts_each_chain, num_chains, labels_anchors, answer_gt_short, anchors_idx_x, num_all_thoughts

def plot_chain_animation(num_chains, num_thoughts_each_chain, coordinates_2d, anchors_idx_x, labels_anchors, answer_gt_short):
    # Extract the lines based on num_thoughts_each_chain
    lines = []
    for chain_idx in range(num_chains):
        start_idx = sum(num_thoughts_each_chain[:chain_idx])
        end_idx = sum(num_thoughts_each_chain[:chain_idx + 1])
        x = list(coordinates_2d[start_idx:end_idx, 0])
        y = list(coordinates_2d[start_idx:end_idx, 1])
        lines.append((x, y))

    # Calculate the global max x and y values
    max_x = max(max(x) for x, _ in lines) + 5
    max_y = max(max(y) for _, y in lines) + 5

    min_x = min(min(x) for x, _ in lines) - 5
    min_y = min(min(y) for _, y in lines) - 5

    # Determine the maximum step count
    max_steps = max(len(x) for x, _ in lines)

    # Create a base figure
    fig = go.Figure()

    # Add initial traces for each line
    for i, (x, y) in enumerate(lines):
        normalized_indices = np.linspace(0, 1, len(x))
        
        fig.add_trace(go.Scatter(
            x=[x[0]], y=[y[0]],
            mode='markers',
            name=f'Chain {i+1}',
            line=dict(width=2),
            marker=dict(
                size=5, 
                color=normalized_indices,  # Color based on index within the chain
                colorscale='RdYlGn',  # Choose a colorscale (Viridis is an example)
                showscale=True,
            ), # green or red
            showlegend=False,
        ))

    colors = px.colors.qualitative.Plotly
    for idx, anchor_name in enumerate(labels_anchors):
        if anchor_name == 'Start': # start
            marker_symbol = 'star'
        elif anchor_name == answer_gt_short: 
            # correct answer
            marker_symbol='diamond'
        else: # negative answer 
            marker_symbol = 'x'

        fig.add_trace(
            go.Scatter(
                x=[coordinates_2d[anchors_idx_x[idx], 0]], y=[coordinates_2d[anchors_idx_x[idx], 1]], 
                mode='markers',
                name=anchor_name, marker=dict(symbol=marker_symbol, size=20, line_width=1, color=colors[idx % len(colors)])
        ))


    # Create frames
    frames = []
    for step in range(1, max_steps + 1):
        frame_data = []
        for i, (x, y) in enumerate(lines):
            # Limit the subset to the length of the current line
            if step <= len(x):
                frame_data.append(go.Scatter(
                    x=x[:step], y=y[:step],
                    mode='markers',
                    name=f'Chain {i+1}',
                    line=dict(width=2),
                    showlegend=False,
                ))
            else:
                # If current step exceeds line length, plot the entire line
                frame_data.append(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    name=f'Chain {i+1}',
                    line=dict(width=2),
                    showlegend=False,
                ))
        
        # Include anchors in every frame
        colors = px.colors.qualitative.Plotly
        for idx, anchor_name in enumerate(labels_anchors):
            if anchor_name == 'Start': # start
                marker_symbol = 'star'
            elif anchor_name == answer_gt_short: 
                # correct answer
                marker_symbol='diamond'
            else: # negative answer 
                marker_symbol = 'x'

            frame_data.append(
                go.Scatter(
                    x=[coordinates_2d[anchors_idx_x[idx], 0]], y=[coordinates_2d[anchors_idx_x[idx], 1]], 
                    mode='markers',
                    name=anchor_name, marker=dict(symbol=marker_symbol, size=20, line_width=1, color=colors[idx % len(colors)])
            ))
            fig.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ), margin=dict(l=10, r=10, t=20, b=10),)

        frames.append(go.Frame(data=frame_data, name=str(step)))

    fig.frames = frames

    # Set layout with sliders, buttons, and dynamic axis ranges
    fig.update_layout(
        xaxis=dict(range=[min_x, max_x]),  # Dynamic x-axis range
        yaxis=dict(range=[min_y, max_y]),  # Dynamic y-axis range
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate', fromcurrent=True)]
                    ),
                ],
            ),
        ],
        sliders=[dict(steps=[dict(method='animate',
                                args=[[str(k)], dict(mode='immediate', frame=dict(duration=500, redraw=True), fromcurrent=True)],
                                label=str(k)) for k in range(1, max_steps + 1)],
                    transition=dict(duration=0))],
    )
    return fig

def plot_single_chain(num_chains, num_thoughts_each_chain, coordinates_2d, coordinates_value, all_thoughts, chain_corr, answer_gt_short, anchors_idx_x, labels_anchors):
    # plot states
    plt.figure(figsize=(8, 6))    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Light24
    for chain_idx in range(num_chains):
        start_idx, end_idx = sum(num_thoughts_each_chain[:chain_idx]), sum(num_thoughts_each_chain[:chain_idx+1])
        x, y = list(coordinates_2d[start_idx:end_idx, 0]), list(coordinates_2d[start_idx:end_idx, 1])

        
        # Normalized indices to use for the color scale
        normalized_indices = np.linspace(0, 1, len(x))
        # if not len(x): continue # eliminate empty chain 
        if len(normalized_indices) == 1: continue
        
        # Create the scatter plot for the original xy points
        fig.add_trace(
            go.Scatter(x=x, y=y,
                            mode='markers', 
                            marker_symbol='diamond' if chain_corr[chain_idx] else 'x',
                            marker_size=10,
                            marker=dict(
                                size=5, 
                                color=normalized_indices,  # Color based on index within the chain
                                colorscale='RdYlGn',  # Choose a colorscale (Viridis is an example)
                                showscale=True,
                            ), # green or red
                            showlegend=False,
                            line_color='black',
                            customdata=[[chain_idx, round(coordinates_value[start_idx+thought_idx], 3), all_thoughts[start_idx+thought_idx]] for thought_idx in range(len(x))],
                            hovertemplate=
                            "<b>Chain-%{customdata[0]}</b><br>"   # Chain index
                            +"PPL: %{customdata[1]}<br>"      # Average PPL
                            # +"Thought: %{customdata[2]}<br>"     # Thought 
                            +"X: %{x}<br>"                       # X value
                            +"Y: %{y}<br>"                         # Y value
                    ))

    # Plot anchors
    #####################################
    colors = px.colors.qualitative.Plotly
    for idx, anchor_name in enumerate(labels_anchors):
        if anchor_name == 'Start': # start
            marker_symbol = 'star'
        elif anchor_name == answer_gt_short: 
            # correct answer
            marker_symbol='diamond'
        else: # negative answer 
            marker_symbol = 'x'

        fig.add_trace(
            go.Scatter(
                x=[coordinates_2d[anchors_idx_x[idx], 0]], y=[coordinates_2d[anchors_idx_x[idx], 1]], 
                mode='markers',
                name=anchor_name, marker=dict(symbol=marker_symbol, size=20, line_width=1, color=colors[idx % len(colors)])
        ))

        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ), margin=dict(l=10, r=10, t=20, b=10),)
    return fig

