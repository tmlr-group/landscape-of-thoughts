"""
Animation visualization utilities for LOT.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Any, Optional

def plot_chain_animation(
    coordinates_2d: np.ndarray,
    num_thoughts_each_chain: List[int],
    num_chains: int,
    anchors_idx_x: List[int],
    labels_anchors: List[str],
    answer_gt_short: str,
    use_contour: bool = True,
    num_frames: int = 20,
    unified_contours: bool = False
) -> go.Figure:
    """
    Create an animated plot of reasoning chains.
    
    Args:
        coordinates_2d (np.ndarray): 2D coordinates for plotting.
        num_thoughts_each_chain (List[int]): Number of thoughts in each chain.
        num_chains (int): Number of chains.
        anchors_idx_x (List[int]): Indices of anchors.
        labels_anchors (List[str]): Labels for anchors.
        answer_gt_short (str): Correct answer.
        use_contour (bool): Whether to use Histogram2dContour for visualization.
        num_frames (int): Number of frames for the animation.
        unified_contours (bool): Whether to unify correct and incorrect contours in a single visualization.
        
    Returns:
        go.Figure: Plotly figure object with animation.
    """
    # Calculate total number of thoughts
    num_all_thoughts = sum(num_thoughts_each_chain)
    
    # Extract the lines based on num_thoughts_each_chain
    correct_lines = []
    wrong_lines = []
    
    for chain_idx in range(num_chains):
        start_idx = sum(num_thoughts_each_chain[:chain_idx])
        end_idx = sum(num_thoughts_each_chain[:chain_idx + 1])
        
        if end_idx <= start_idx:
            continue
            
        x = list(coordinates_2d[start_idx:end_idx, 0])
        y = list(coordinates_2d[start_idx:end_idx, 1])
        
        if len(x) <= 1:
            continue
            
        # Determine if this is a correct or wrong chain based on the answer
        # We need to check if this chain's answer matches the ground truth
        chain_is_correct = False
        
        # In the anchors_idx_x, the first index after all thoughts is the "Start" anchor
        # and the rest are the answer options
        answer_indices = [i for i in range(len(anchors_idx_x)) if anchors_idx_x[i] >= num_all_thoughts]
        
        # The chain is correct if its final point is closer to the correct answer
        if answer_indices and len(answer_indices) > 1:  # We have at least Start and one answer
            # Find the index of the correct answer in labels_anchors
            correct_answer_idx = labels_anchors.index(answer_gt_short) if answer_gt_short in labels_anchors else -1
            
            if correct_answer_idx >= 0:
                # Check if this chain's answer is correct
                chain_is_correct = (chain_idx < len(num_thoughts_each_chain) and 
                                   correct_answer_idx == chain_idx % len(answer_indices))
        
        if chain_is_correct:
            correct_lines.append((x, y))
        else:
            wrong_lines.append((x, y))

    # Calculate the global max x and y values with padding
    all_lines = correct_lines + wrong_lines
    max_x = max(max(x) for x, _ in all_lines if x) + 5 if any(x for x, _ in all_lines) else 5
    max_y = max(max(y) for _, y in all_lines if y) + 5 if any(y for _, y in all_lines) else 5
    min_x = min(min(x) for x, _ in all_lines if x) - 5 if any(x for x, _ in all_lines) else -5
    min_y = min(min(y) for _, y in all_lines if y) - 5 if any(y for _, y in all_lines) else -5

    # Create a base figure
    fig = go.Figure()

    # Add anchor points
    colors = px.colors.qualitative.Plotly
    for idx, anchor_name in enumerate(labels_anchors):
        if anchor_name == 'Start':  # start
            marker_symbol = 'star'
            marker_color = colors[idx % len(colors)]
        elif anchor_name == answer_gt_short:  # correct answer
            marker_symbol = 'diamond'
            marker_color = "green"
        else:  # negative answer 
            marker_symbol = 'x'
            marker_color = "red"

        fig.add_trace(
            go.Scatter(
                x=[coordinates_2d[anchors_idx_x[idx], 0]], 
                y=[coordinates_2d[anchors_idx_x[idx], 1]], 
                mode='markers',
                name=anchor_name, 
                marker=dict(
                    symbol=marker_symbol, 
                    size=20, 
                    line_width=1, 
                    color=marker_color
                ),
                showlegend=True
            )
        )

    # Create frames for animation
    frames = []
    
    if use_contour:
        # For contour-based animation
        for frame_idx in range(num_frames):
            frame_data = []
            
            # Add anchor points to each frame
            for idx, anchor_name in enumerate(labels_anchors):
                if anchor_name == 'Start':  # start
                    marker_symbol = 'star'
                    marker_color = colors[idx % len(colors)]
                elif anchor_name == answer_gt_short:  # correct answer
                    marker_symbol = 'diamond'
                    marker_color = "green"
                else:  # negative answer 
                    marker_symbol = 'x'
                    marker_color = "red"

                frame_data.append(
                    go.Scatter(
                        x=[coordinates_2d[anchors_idx_x[idx], 0]], 
                        y=[coordinates_2d[anchors_idx_x[idx], 1]], 
                        mode='markers',
                        name=anchor_name, 
                        marker=dict(
                            symbol=marker_symbol, 
                            size=20, 
                            line_width=1, 
                            color=marker_color
                        ),
                        showlegend=True
                    )
                )
            
            # Calculate progress for this frame (0 to 1)
            progress = frame_idx / (num_frames - 1) if num_frames > 1 else 1
            
            # Process wrong chains
            wrong_x = []
            wrong_y = []
            wrong_weights = []
            
            for x, y in wrong_lines:
                # Calculate how many points to include based on progress
                points_to_include = max(1, int(len(x) * progress))
                
                # Add points and their weights
                wrong_x.extend(x[:points_to_include])
                wrong_y.extend(y[:points_to_include])
                
                # Create weights that represent progress within each chain
                chain_weights = np.linspace(0, 1, len(x))
                wrong_weights.extend(chain_weights[:points_to_include])
            
            # Process correct chains
            correct_x = []
            correct_y = []
            correct_weights = []
            
            for x, y in correct_lines:
                # Calculate how many points to include based on progress
                points_to_include = max(1, int(len(x) * progress))
                
                # Add points and their weights
                correct_x.extend(x[:points_to_include])
                correct_y.extend(y[:points_to_include])
                
                # Create weights that represent progress within each chain
                chain_weights = np.linspace(0, 1, len(x))
                correct_weights.extend(chain_weights[:points_to_include])
            
            if unified_contours:
                # Combine all points for unified visualization
                all_x = wrong_x + correct_x
                all_y = wrong_y + correct_y
                all_weights = []
                
                if wrong_weights:
                    all_weights.extend(wrong_weights)
                if correct_weights:
                    all_weights.extend(correct_weights)
                
                if all_x:
                    all_weights = np.array(all_weights)
                    
                    # Calculate thresholds for segments
                    all_thresholds = np.percentile(all_weights, [20, 40, 60, 80]) if len(all_weights) > 0 else np.array([0.2, 0.4, 0.6, 0.8])
                    
                    # Create 5 segments
                    for i in range(5):
                        if i == 0:
                            all_mask = all_weights <= all_thresholds[0] if len(all_weights) > 0 else np.array([], dtype=bool)
                        elif i == 4:
                            all_mask = all_weights > all_thresholds[3] if len(all_weights) > 0 else np.array([], dtype=bool)
                        else:
                            if len(all_weights) > 0:
                                all_mask = (all_weights > all_thresholds[i-1]) & (all_weights <= all_thresholds[i])
                            else:
                                all_mask = np.array([], dtype=bool)
                        
                        # Get segment data
                        if len(all_weights) > 0:
                            all_x_segment = np.array(all_x)[all_mask]
                            all_y_segment = np.array(all_y)[all_mask]
                        else:
                            all_x_segment = np.array([])
                            all_y_segment = np.array([])
                        
                        # Add contour if there are points
                        if len(all_x_segment) > 0:
                            frame_data.append(
                                go.Histogram2dContour(
                                    x=all_x_segment,
                                    y=all_y_segment,
                                    colorscale="Viridis",
                                    showscale=True,
                                    histfunc='count',
                                    contours=dict(
                                        showlines=True,
                                        coloring='fill'
                                    ),
                                    colorbar=dict(
                                        title=dict(
                                            text="Progress",
                                            side="right"
                                        ),
                                        thickness=20,
                                        len=0.6,
                                        y=0.5
                                    ),
                                    autocontour=True,
                                    opacity=0.7,
                                    name=f'Reasoning Range {i+1}'
                                )
                            )
                        else:
                            # Add empty scatter to maintain structure
                            frame_data.append(
                                go.Scatter(
                                    x=[],
                                    y=[],
                                    mode='markers',
                                    showlegend=False,
                                    name=f'Reasoning Range {i+1}'
                                )
                            )
            else:
                # Create contour segments for wrong chains
                if wrong_x:
                    wrong_weights = np.array(wrong_weights)
                    
                    # Calculate thresholds for segments
                    wrong_thresholds = np.percentile(wrong_weights, [20, 40, 60, 80]) if len(wrong_weights) > 0 else np.array([0.2, 0.4, 0.6, 0.8])
                    
                    # Create 5 segments
                    for i in range(5):
                        if i == 0:
                            wrong_mask = wrong_weights <= wrong_thresholds[0] if len(wrong_weights) > 0 else np.array([], dtype=bool)
                        elif i == 4:
                            wrong_mask = wrong_weights > wrong_thresholds[3] if len(wrong_weights) > 0 else np.array([], dtype=bool)
                        else:
                            if len(wrong_weights) > 0:
                                wrong_mask = (wrong_weights > wrong_thresholds[i-1]) & (wrong_weights <= wrong_thresholds[i])
                            else:
                                wrong_mask = np.array([], dtype=bool)
                        
                        # Get segment data
                        if len(wrong_weights) > 0:
                            wrong_x_segment = np.array(wrong_x)[wrong_mask]
                            wrong_y_segment = np.array(wrong_y)[wrong_mask]
                        else:
                            wrong_x_segment = np.array([])
                            wrong_y_segment = np.array([])
                        
                        # Add contour if there are points
                        if len(wrong_x_segment) > 0:
                            frame_data.append(
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
                                )
                            )
                        else:
                            # Add empty scatter to maintain structure
                            frame_data.append(
                                go.Scatter(
                                    x=[],
                                    y=[],
                                    mode='markers',
                                    showlegend=False,
                                    name=f'Wrong Range {i+1}'
                                )
                            )
                
                # Create contour segments for correct chains
                if correct_x:
                    correct_weights = np.array(correct_weights)
                    
                    # Calculate thresholds for segments
                    correct_thresholds = np.percentile(correct_weights, [20, 40, 60, 80]) if len(correct_weights) > 0 else np.array([0.2, 0.4, 0.6, 0.8])
                    
                    # Create 5 segments
                    for i in range(5):
                        if i == 0:
                            correct_mask = correct_weights <= correct_thresholds[0] if len(correct_weights) > 0 else np.array([], dtype=bool)
                        elif i == 4:
                            correct_mask = correct_weights > correct_thresholds[3] if len(correct_weights) > 0 else np.array([], dtype=bool)
                        else:
                            if len(correct_weights) > 0:
                                correct_mask = (correct_weights > correct_thresholds[i-1]) & (correct_weights <= correct_thresholds[i])
                            else:
                                correct_mask = np.array([], dtype=bool)
                        
                        # Get segment data
                        if len(correct_weights) > 0:
                            correct_x_segment = np.array(correct_x)[correct_mask]
                            correct_y_segment = np.array(correct_y)[correct_mask]
                        else:
                            correct_x_segment = np.array([])
                            correct_y_segment = np.array([])
                        
                        # Add contour if there are points
                        if len(correct_x_segment) > 0:
                            frame_data.append(
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
                                )
                            )
                        else:
                            # Add empty scatter to maintain structure
                            frame_data.append(
                                go.Scatter(
                                    x=[],
                                    y=[],
                                    mode='markers',
                                    showlegend=False,
                                    name=f'Correct Range {i+1}'
                                )
                            )
            
            frames.append(go.Frame(data=frame_data, name=str(frame_idx)))
    else:
        # For traditional line-based animation
        max_steps = max((len(x) for x, _ in all_lines), default=1)
        
        for step in range(1, max_steps + 1):
            frame_data = []
            
            # Add wrong chains
            for i, (x, y) in enumerate(wrong_lines):
                if not x or not y:
                    continue
                    
                # Normalized indices for color
                normalized_indices = np.linspace(0, 1, len(x))
                
                # Limit the subset to the length of the current line
                if step <= len(x):
                    frame_data.append(go.Scatter(
                        x=x[:step], y=y[:step],
                        mode='markers+lines',
                        name=f'Wrong Chain {i+1}',
                        line=dict(width=2, color='rgba(255,0,0,0.3)'),
                        marker=dict(
                            size=8,
                            color=normalized_indices[:step],
                            colorscale='Reds',
                            showscale=False
                        ),
                        showlegend=True,
                    ))
                else:
                    # If current step exceeds line length, plot the entire line
                    frame_data.append(go.Scatter(
                        x=x, y=y,
                        mode='markers+lines',
                        name=f'Wrong Chain {i+1}',
                        line=dict(width=2, color='rgba(255,0,0,0.3)'),
                        marker=dict(
                            size=8,
                            color=normalized_indices,
                            colorscale='Reds',
                            showscale=False
                        ),
                        showlegend=True,
                    ))
            
            # Add correct chains
            for i, (x, y) in enumerate(correct_lines):
                if not x or not y:
                    continue
                    
                # Normalized indices for color
                normalized_indices = np.linspace(0, 1, len(x))
                
                # Limit the subset to the length of the current line
                if step <= len(x):
                    frame_data.append(go.Scatter(
                        x=x[:step], y=y[:step],
                        mode='markers+lines',
                        name=f'Correct Chain {i+1}',
                        line=dict(width=2, color='rgba(0,0,255,0.3)'),
                        marker=dict(
                            size=8,
                            color=normalized_indices[:step],
                            colorscale='Blues',
                            showscale=False
                        ),
                        showlegend=True,
                    ))
                else:
                    # If current step exceeds line length, plot the entire line
                    frame_data.append(go.Scatter(
                        x=x, y=y,
                        mode='markers+lines',
                        name=f'Correct Chain {i+1}',
                        line=dict(width=2, color='rgba(0,0,255,0.3)'),
                        marker=dict(
                            size=8,
                            color=normalized_indices,
                            colorscale='Blues',
                            showscale=False
                        ),
                        showlegend=True,
                    ))
            
            # Include anchors in every frame
            for idx, anchor_name in enumerate(labels_anchors):
                if anchor_name == 'Start':  # start
                    marker_symbol = 'star'
                    marker_color = colors[idx % len(colors)]
                elif anchor_name == answer_gt_short:  # correct answer
                    marker_symbol = 'diamond'
                    marker_color = "green"
                else:  # negative answer 
                    marker_symbol = 'x'
                    marker_color = "red"

                frame_data.append(
                    go.Scatter(
                        x=[coordinates_2d[anchors_idx_x[idx], 0]], 
                        y=[coordinates_2d[anchors_idx_x[idx], 1]], 
                        mode='markers',
                        name=anchor_name, 
                        marker=dict(
                            symbol=marker_symbol, 
                            size=20, 
                            line_width=1, 
                            color=marker_color
                        ),
                        showlegend=True
                    )
                )

            frames.append(go.Frame(data=frame_data, name=str(step)))

    fig.frames = frames

    # Set layout with sliders, buttons, and dynamic axis ranges
    fig.update_layout(
        xaxis=dict(
            range=[min_x, max_x],
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
        ),
        yaxis=dict(
            range=[min_y, max_y],
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(frame=dict(duration=200, redraw=True), fromcurrent=True)]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate', fromcurrent=True)]
                    ),
                ],
                x=0.1,
                y=1.1,
                xanchor='right',
                yanchor='top'
            ),
        ],
        sliders=[dict(
            steps=[dict(
                method='animate',
                args=[[str(k)], dict(mode='immediate', frame=dict(duration=200, redraw=True), fromcurrent=True)],
                label=str(k)
            ) for k in range(len(frames))],
            transition=dict(duration=0),
            x=0.1,
            y=1.15,
            currentvalue=dict(
                font=dict(size=12),
                prefix='Step: ',
                visible=True,
                xanchor='right'
            ),
            len=0.9
        )],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=10, r=10, t=100, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color="black"),
        height=600,
        width=900,
        title="Reasoning Trace Animation"
    )

    # Apply template for consistent styling
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
                gridcolor="lightgray",
                zerolinecolor="black",
            ),
            
            yaxis=dict(
                title_font_color="black", 
                tickfont_color="black",
                linecolor="black",
                gridcolor="lightgray",
                zerolinecolor="black",
            ),
            
            hoverlabel=dict(
                font_color="black",
                bgcolor="white"
            ),
            
            annotations=[dict(font_color="black")],
            shapes=[dict(line_color="black")],
        )
    )

    fig.update_layout(template=template)
    
    return fig


def create_animation_from_file(
    thoughts_file: str,
    distance_matrix_file: str = None,
    tool: str = 'tsne',
    use_contour: bool = True,
    num_frames: int = 20,
    unified_contours: bool = False
) -> go.Figure:
    """
    Create an animation from a thoughts file and distance matrix file.
    
    Args:
        thoughts_file (str): Path to the thoughts file.
        distance_matrix_file (str): Path to the distance matrix file. If None, inferred from thoughts_file.
        tool (str): Tool to use for dimensionality reduction ('tsne' or 'umap').
        use_contour (bool): Whether to use Histogram2dContour for visualization.
        num_frames (int): Number of frames for the animation.
        unified_contours (bool): Whether to unify correct and incorrect contours in a single visualization.
        
    Returns:
        go.Figure: Plotly figure object with animation.
    """
    import json
    import pickle as pkl
    import os
    import copy
    from sklearn.manifold import TSNE
    
    # Try to infer distance_matrix_file if not provided
    if distance_matrix_file is None:
        distance_matrix_file = thoughts_file.replace(".json", ".pkl")
        distance_matrix_file = distance_matrix_file.replace("thoughts/", "distance_matrix/")
    
    # Check if files exist
    if not os.path.exists(thoughts_file):
        raise FileNotFoundError(f"Thoughts file not found: {thoughts_file}")
    if not os.path.exists(distance_matrix_file):
        raise FileNotFoundError(f"Distance matrix file not found: {distance_matrix_file}")
    
    # Load data
    with open(thoughts_file, 'r') as f:
        trial_data = json.load(f)
    
    with open(distance_matrix_file, 'rb') as f:
        distance_matrix = pkl.load(f)
    
    # Extract data
    model_input = trial_data["model_input"]
    answers = trial_data["answers"]
    answer_gt_short = trial_data["answer_gt_short"]
    trial_thoughts = trial_data["trial_thoughts"]
    
    # Parse thoughts
    num_chains = len(trial_thoughts)
    num_thoughts_each_chain = [len(thoughts) for thoughts, _, _ in trial_thoughts]
    all_thoughts = []
    for thoughts, _, _ in trial_thoughts:
        all_thoughts.extend(thoughts)
    all_thoughts = np.array(all_thoughts)
    num_all_thoughts = len(all_thoughts)
    
    # Set up anchors
    anchors = [model_input] + answers  # question and answers
    num_anchors = len(anchors)
    anchors_idx_y = [i for i in range(num_anchors)]
    anchors_idx_x = [(num_all_thoughts + i) for i in range(num_anchors)]
    
    # Determine labels based on dataset
    if "strategyqa" in thoughts_file:
        labels_anchors = ["Start", 'yes', 'no']
        if answer_gt_short:
            answer_gt_short = 'yes'
        else:
            answer_gt_short = 'no'
    elif "mmlu" in thoughts_file:
        labels_anchors = ["Start", 'A', 'B', 'C', 'D']
    else:
        labels_anchors = ["Start", 'A', 'B', 'C', 'D', 'E']
    
    gt_idx = labels_anchors.index(answer_gt_short)
    
    # Normalize the distance matrix
    processed_distance_matrix = copy.deepcopy(distance_matrix)
    
    # Normalize thought (T matrix)
    for chain_idx in range(num_chains): 
        start_idx, end_idx = sum(num_thoughts_each_chain[:chain_idx]), sum(num_thoughts_each_chain[:chain_idx+1])
        for thought_idx in range(start_idx, end_idx):
            for anchor_idx in range(1, num_anchors):
                processed_distance_matrix[thought_idx, anchor_idx] = distance_matrix[thought_idx, anchor_idx] / np.sum(distance_matrix[thought_idx, anchors_idx_y[1:]])
    
    # Normalize answer (A matrix)
    A = processed_distance_matrix[num_all_thoughts:]
    A[np.diag_indices(A.shape[0])] = 0
    normed_A = copy.deepcopy(A)
    for col_idx in range(1, num_anchors):
        normed_A[0][col_idx] = 1 + A[0][col_idx] / np.sum(A[0, anchors_idx_y[1:]])
    normed_A[:, 0] = normed_A[0, :]  # copy same elements to 0-th col
    normed_A[1:, 1:] = 1 / (num_anchors-1) 
    normed_A[np.diag_indices(normed_A.shape[0])] = 0
    processed_distance_matrix[num_all_thoughts:] = normed_A
    
    # Generate 2D coordinates
    if tool == 'tsne':
        tsne = TSNE(n_components=2, perplexity=10, random_state=42)
        coordinates_2d = tsne.fit_transform(processed_distance_matrix[:, 1:])
    elif tool == 'umap':
        import umap
        coordinates_2d = umap.UMAP(n_neighbors=30, min_dist=0.25, n_components=2, metric='dice', random_state=42).fit_transform(processed_distance_matrix)
    else:
        raise ValueError(f"Unknown tool: {tool}")
    
    # Create animation
    fig = plot_chain_animation(
        coordinates_2d=coordinates_2d,
        num_thoughts_each_chain=num_thoughts_each_chain,
        num_chains=num_chains,
        anchors_idx_x=anchors_idx_x,
        labels_anchors=labels_anchors,
        answer_gt_short=answer_gt_short,
        use_contour=use_contour,
        num_frames=num_frames,
        unified_contours=unified_contours
    )
    
    return fig 