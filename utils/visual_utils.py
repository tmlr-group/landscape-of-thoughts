import copy
import json
import os
import pickle as pkl

import igraph as ig
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap.umap_ as umap
from matplotlib.colors import to_hex
from sklearn.manifold import TSNE
from tqdm import tqdm

from algorithms import *
from benchmarks import *
from models import *

plt.style.use('default')

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.level = 0
        self.id = value
        self.parent = None
    
    def add_child(self, child):
        self.children.append(child)
        child.level = self.level + 1
        child.parent = self

def build_tree_from_cots(cots_list):
    root = Node("Root")
    level_nodes = {}
    
    for cot in cots_list:
        current_parent = root
        
        for i, sentence in enumerate(cot):
            new_node = Node(sentence)
            
            if i not in level_nodes:
                level_nodes[i] = []
            
            similar_node = None
            for existing_node in level_nodes[i]:
                if existing_node.value == sentence:
                    similar_node = existing_node
                    break
            
            if similar_node:
                current_parent = similar_node
            else:
                current_parent.add_child(new_node)
                level_nodes[i].append(new_node)
                current_parent = new_node

    return root

def tree_to_json(node):
    json_list = []
    
    def traverse(node):
        node_dict = {
            "id": node.id,
            "parent": node.parent.id if node.parent else None,
            "distance": node.distance,
        }
        json_list.append(node_dict)
        
        for child in node.children:
            traverse(child)
    
    traverse(node)
    return json_list # json.dumps(json_list, indent=2)



def tree_to_dataframe(node):
    rows = []
    node_id_counter = 0
    node_id_map = {}  # Map to store node value to id mapping
    
    def traverse(node, level=0):
        nonlocal node_id_counter
        
        # Assign ID to current node and store in map
        if node.id not in node_id_map:
            node_id_map[node.id] = node_id_counter
            current_id = node_id_counter
            node_id_counter += 1
        else:
            current_id = node_id_map[node.id]
        
        # Get parent ID
        parent_id = node_id_map.get(node.parent.id, "NaN") if node.parent else "NaN"
        
        # Create row dictionary
        print(node.distance)
        row = {
            'id': current_id,
            'distance': node.distance[1:] if len(node.distance) else [], 
            'parent': parent_id,
            'level': level  # Add level information
        }

        rows.append(row)
        
        # Process children with incremented level
        for child in node.children:
            traverse(child, level + 1)
    
    traverse(node)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder columns to match desired format
    df = df[['id', 'distance', 'parent', 'level']]
    
    return df


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

def load_chain_data(thoughts_file: str = "None",
        tool: str = 'tsne',
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

    return distance_matrix, num_chains, num_thoughts_each_chain, coordinates_2d, normed_A, all_answers, answer_gt_short

def get_sample_distance_matrix(
        model="Meta-Llama-3.1-70B-Instruct-Turbo", 
        dataset='aqua', 
        method='zero_shot_cot',
        root='exp-data-scale',
    ):
    # load thoughts
    # #######################################
    datas = {}
    if dataset == "mmlu":
        target_A_matrix = np.ones((4,4)) * (1/4) 
    elif dataset == "strategyqa":
        target_A_matrix = np.ones((2,2)) * (1/3) 
    else:
        target_A_matrix = np.ones((5,5)) * (1/4) 
    target_A_matrix[np.diag_indices(target_A_matrix.shape[0])] = 0

    for sample_idx in tqdm(range(50), ncols=50):
        if method in ["mcts", "tot"]:
            file_path = f'./exp-data-searching/{dataset}/thoughts/{model}--{method}--{dataset}--{sample_idx}.json'
        else:
            file_path = f'./{root}/{dataset}/thoughts/{model}--{method}--{dataset}--{sample_idx}.json'
        (distance_matrix, num_chains, num_thoughts_each_chain, coordinates_2d, _, all_answers, answer_gt_short) = load_chain_data(thoughts_file=file_path)
        
        datas[sample_idx] = {
            "distance_matrix": distance_matrix,
            "coordinates_2d": coordinates_2d,
            "num_thoughts_each_chain": num_thoughts_each_chain,
            "num_chains": num_chains,
            "all_answers": all_answers,
            "answer_gt_short": answer_gt_short,
        }

    return datas


def load_thoughts_to_json(model="Meta-Llama-3.1-70B-Instruct-Turbo", dataset='aqua', method='zero_shot_cot',):
    
    datas = get_sample_distance_matrix(model=model, dataset=dataset, method=method,)

    all_sample_chains = []
    for _, plot_data in datas.items():
        all_thoughts, num_thoughts_each_chain, num_chains, _, _, _ = plot_data.values()
        # Collect points for each chain
        sample_chains = []
        for chain_idx in range(num_chains):
            start_idx = sum(num_thoughts_each_chain[:chain_idx])
            end_idx = sum(num_thoughts_each_chain[:chain_idx+1])
            if end_idx <= start_idx:
                continue
            chain_thoughts = all_thoughts[start_idx:end_idx]
            sample_chains.append(chain_thoughts)

        all_sample_chains.append(sample_chains)

    all_json_thoughts = []
    all_df_thoughts = []
    for sample_chains in all_sample_chains:
        root = build_tree_from_cots(sample_chains)
        json_thoughts = tree_to_json(root)
        df_thoughts = tree_to_dataframe(root)

        all_json_thoughts.append(json_thoughts)
        all_df_thoughts.append(df_thoughts)

    return all_json_thoughts, all_df_thoughts

def matplotlib_to_plotly_tree(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale

def ploty_igraph_tree(data, model_name, prompt_method='ToT', show_fig=True, save_fig=False, fig_name='demo.pdf'):
    # Create a new directed graph
    g = ig.Graph(directed=True)

    # Track all child nodes to help find the root
    children = set()

    # Add vertices and track children
    for item in data:
        g.add_vertex(name=item['id'], value=item['score'])
        if item['parent']:
            children.add(item['id'])

    # Add edges based on the parent-child relationships and find the root
    root_candidates = []
    for item in data:
        if item['parent']:
            g.add_edge(item['parent'], item['id'])
        if item['id'] not in children:
            root_candidates.append(item['id'])


    g.simplify(multiple=True, loops=True)
    g.vs.select(_degree=0).delete()

    nr_vertices = g.vcount()
    labels = g.vs['name']
    values = g.vs['value']

    root = root_candidates[0] if root_candidates else None
    lay = g.layout_reingold_tilford(root=[g.vs.find(name=root).index])

    position = {k: lay[k] for k in range(nr_vertices)}
    Y = [lay[k][1] for k in range(nr_vertices)]
    M = max(Y)

    E = [e.tuple for e in g.es] # list of edges

    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2*M-position[k][1] for k in range(L)]
    Xe = []
    Ye = []
    for edge in E:
        Xe+=[position[edge[0]][0],position[edge[1]][0], None]
        Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]


    minima = min(values)
    maxima = max(values)

    color_map = cm.coolwarm

    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=color_map)

    color_map = matplotlib_to_plotly_tree(color_map, 255)

    node_colors = []

    for v in values:
        node_colors.append(to_hex(mapper.to_rgba(v)))


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=Xe,
                    y=Ye,
                    mode='lines',
                    name='Steps',
                    line=dict(color='rgb(210,210,210)', width=1),
                    hoverinfo='none'
                    ))
    fig.add_trace(go.Scatter(x=Xn,
                    y=Yn,
                    mode='markers',
                    name='Thoughts',
                    marker=dict(symbol='circle-dot',
                                    size=18,
                                    color=node_colors, 
                                    line=dict(color='rgb(50,50,50)', width=1)
                                    ),
                    text=labels,
                    hoverinfo='text',
                    opacity=0.8
                    ))

    colorbar_trace  = go.Scatter(x=[None],
                                y=[None],
                                mode='markers',
                                marker=dict(
                                    colorscale=color_map, 
                                    showscale=True,
                                    cmin=minima,
                                    cmax=maxima,
                                    colorbar=dict(thickness=5, tickvals=[minima, maxima], ticktext=['Low', 'High'], outlinewidth=0)
                                ),
                                hoverinfo='none'
                                )

    fig['layout']['showlegend'] = False
    fig.add_trace(colorbar_trace)

    fig.update_layout(
        title=f'{prompt_method} Game24 {model_name}',
        font_family="Gill Sans MT",
        font_color="Black",
        title_font_family="Gill Sans MT",
        title_font_color="Black",
        title_font_size=40,
        legend_title_font_color="green"
    )
    fig.update_xaxes(title_font_family="Gill Sans MT")

    if show_fig:
        fig.show()

    if save_fig:
        import plotly.io as pio 
        pio.write_image(
            fig, f'results/{model_name}/{prompt_method}/{fig_name}',
            scale=6, width=1080, height=608
        )

    return g

def plot_means_with_std(all_means):
    # Get unique levels and methods
    all_levels = sorted(set(level for level_means in all_means.values() for level in level_means.keys()))
    methods = list(all_means.keys())
    
    # Define colors for each method
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 
              'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)']
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each method
    for idx, method_name in enumerate(methods):
        color = colors[idx % len(colors)]  # Cycle through colors if more methods than colors
        means = [np.mean(all_means[method_name].get(level, [0])) for level in all_levels]
        stds = [np.std(all_means[method_name].get(level, [0])) for level in all_levels]
        
        # Add main line and markers
        fig.add_trace(go.Scatter(
            x=all_levels,
            y=means,
            name=method_name,
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(color=color, size=8),
            hovertemplate="Level: %{x}<br>" +
                         "Mean: %{y:.4f}<br>" +
                         "Method: " + method_name
        ))
        
        # Add error bands with same color
        fig.add_trace(go.Scatter(
            x=all_levels + all_levels[::-1],
            y=[m + s for m, s in zip(means, stds)] + [m - s for m, s in zip(means[::-1], stds[::-1])],
            fill='toself',
            fillcolor=color,
            opacity=0.2,
            line=dict(width=0, color=color),
            name=f'{method_name} (±σ)',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Update layout
    fig.update_layout(
        # title="Mean Values with Standard Deviation by Method and Level",
        xaxis_title="Level",
        yaxis_title="Value",
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            font=dict(size=25)  # Set legend font size to 25
        )
    )
    
    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig

def plot_multiple_means_with_std(list_of_means_with_names, title="Model Comparison Across Levels"):
    """
    Args:
        list_of_means_with_names: list of tuples [(name1, all_means1), (name2, all_means2), ...]
        where each all_means is the dictionary containing method results
    """
    # Get unique levels across all datasets
    all_levels = sorted(set(
        level
        for _, means_dict in list_of_means_with_names
        for level_means in means_dict.values()
        for level in level_means.keys()
    ))
    
    # Define colors for each method
    colors = px.colors.qualitative.Set1
    
    # Create figure
    fig = go.Figure()
    
    # Track used colors for each method to maintain consistency
    method_colors = {}
    color_idx = 0
    
    # Plot each dataset
    for model_name, all_means in list_of_means_with_names:
        for method_name in all_means.keys():
            # Create a unique identifier for the legend
            legend_name = f"{model_name} - {method_name}"
            
            # Ensure same method gets same color across models
            if method_name not in method_colors:
                method_colors[method_name] = colors[color_idx % len(colors)]
                color_idx += 1

            # Get color
            color = colors[color_idx % len(colors)]
            color_idx += 1
            
            # Calculate means and stds
            means = [np.mean(all_means[method_name].get(level, [0])) for level in all_levels]
            stds = [np.std(all_means[method_name].get(level, [0])) for level in all_levels]
            
            # Add main line and markers
            fig.add_trace(go.Scatter(
                x=all_levels,
                y=means,
                name=legend_name,
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(color=color, size=8),
                hovertemplate="Level: %{x}<br>" +
                             "Mean: %{y:.4f}<br>" +
                             "Model: " + model_name + "<br>" +
                             "Method: " + method_name
            ))
            
            # Add error bands
            fig.add_trace(go.Scatter(
                x=all_levels + all_levels[::-1],
                y=[m + s for m, s in zip(means, stds)] + 
                  [m - s for m, s in zip(means[::-1], stds[::-1])],
                fill='toself',
                fillcolor=color,
                opacity=0.2,
                line=dict(width=0, color=color),
                name=f'{legend_name} (±σ)',
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Level",
        yaxis_title="Value",
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            font=dict(size=25)
        )
    )
    
    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig