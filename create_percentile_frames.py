import plotly.graph_objects as go
import numpy as np
import time

def create_percentile_frames(frame_data, coordinates_2d, labels_anchors, num_percentiles=5, connect_anchors=True, anchor_size=24, correct_path_color="green"):
    """
    Create frames based on percentiles of reasoning chain completion.
    
    Args:
        frame_data: Dictionary containing processed data for each frame
        coordinates_2d: 2D coordinates for plotting
        labels_anchors: List of anchor labels
        num_percentiles: Number of percentile ranges to create
        connect_anchors: Whether to connect anchor points with a line to show the correct path (default: True)
        anchor_size: Size of anchor markers (default: 24)
        correct_path_color: Color of the line connecting the correct path anchors (default: "green")
        
    Returns:
        List[go.Figure]: List of static frame figures
    """
    figures = []
    
    # For each percentile range
    for i in range(num_percentiles):
        fig = go.Figure()
        
        # Calculate percentile range
        low_pct = i / num_percentiles
        high_pct = (i + 1) / num_percentiles
        
        # Process each sample in frame_data
        for key, data in frame_data.items():
            num_thoughts_each_chain = data["num_thoughts_each_chain"]
            num_chains = data["num_chains"]
            all_answers = data["all_answers"]
            answer_gt_short = data["answer_gt_short"]
            
            # Separate correct and wrong chains
            correct_points = []
            wrong_points = []
            
            for chain_idx in range(num_chains):
                start_idx = sum(num_thoughts_each_chain[:chain_idx])
                end_idx = sum(num_thoughts_each_chain[:chain_idx + 1])
                
                if end_idx <= start_idx:
                    continue
                
                # Calculate indices for this percentile range
                chain_length = end_idx - start_idx
                low_idx = start_idx + int(chain_length * low_pct)
                high_idx = start_idx + int(chain_length * high_pct)
                
                # Only include points in this percentile range
                if high_idx > low_idx:
                    chain_points = coordinates_2d[low_idx:high_idx]
                    
                    if all_answers[chain_idx] == answer_gt_short:
                        correct_points.append(chain_points)
                    else:
                        wrong_points.append(chain_points)
        
            # Add contours for wrong chains (only for this percentile)
            if wrong_points and len(wrong_points) > 0:
                try:
                    wrong_x = np.concatenate([p[:, 0] for p in wrong_points if len(p) > 0])
                    wrong_y = np.concatenate([p[:, 1] for p in wrong_points if len(p) > 0])
                    if len(wrong_x) > 0:
                        fig.add_trace(
                            go.Histogram2dContour(
                                x=wrong_x,
                                y=wrong_y,
                                colorscale="Reds",
                                showscale=False,
                                histfunc='count',
                                contours=dict(
                                    showlines=True,
                                    coloring='fill'
                                ),
                                autocontour=True,
                                opacity=0.6,
                                name=f'Wrong Chains {int(low_pct*100)}-{int(high_pct*100)}%'
                            )
                        )
                except ValueError:
                    # Skip if there's an issue with concatenation
                    pass
            
            # Add contours for correct chains (only for this percentile)
            if correct_points and len(correct_points) > 0:
                try:
                    correct_x = np.concatenate([p[:, 0] for p in correct_points if len(p) > 0])
                    correct_y = np.concatenate([p[:, 1] for p in correct_points if len(p) > 0])
                    if len(correct_x) > 0:
                        fig.add_trace(
                            go.Histogram2dContour(
                                x=correct_x,
                                y=correct_y,
                                colorscale="Blues",
                                showscale=False,
                                histfunc='count',
                                contours=dict(
                                    showlines=True,
                                    coloring='fill'
                                ),
                                autocontour=True,
                                opacity=0.6,
                                name=f'Correct Chains {int(low_pct*100)}-{int(high_pct*100)}%'
                            )
                        )
                except ValueError:
                    # Skip if there's an issue with concatenation
                    pass
        
        # Collect anchor coordinates for connecting line
        start_anchor_idx = None
        correct_anchor_idx = None
        anchor_x_coords = []
        anchor_y_coords = []
        anchor_names = []
        anchor_colors = []
        
        # Add anchor points
        for idx, anchor_name in enumerate(labels_anchors):
            if idx >= len(coordinates_2d) - len(labels_anchors):
                anchor_idx = len(coordinates_2d) - len(labels_anchors) + idx
                
                # Set marker style based on anchor type
                if anchor_name == 'Start':
                    marker_symbol = 'star'
                    marker_color = 'blue'
                    start_anchor_idx = len(anchor_x_coords)
                elif anchor_name == data.get("answer_gt_short", ""):
                    marker_symbol = 'diamond'
                    marker_color = "green"
                    correct_anchor_idx = len(anchor_x_coords)
                else:
                    marker_symbol = 'x'
                    marker_color = "red"
                
                if anchor_idx < len(coordinates_2d):
                    # Store coordinates for connecting line
                    anchor_x_coords.append(coordinates_2d[anchor_idx, 0])
                    anchor_y_coords.append(coordinates_2d[anchor_idx, 1])
                    anchor_names.append(anchor_name)
                    anchor_colors.append(marker_color)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[coordinates_2d[anchor_idx, 0]],
                            y=[coordinates_2d[anchor_idx, 1]],
                            mode='markers+text',
                            name=anchor_name,
                            text=[anchor_name],
                            textposition="top center",
                            marker=dict(
                                symbol=marker_symbol,
                                size=anchor_size,
                                line_width=2,
                                color=marker_color
                            ),
                            hoverinfo='text',
                            hovertext=anchor_name,
                            showlegend=True
                        )
                    )
        
        # Update layout
        fig.update_layout(
            xaxis=dict(
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
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color="black"),
            height=600,
            width=900,
            title=f"Reasoning Trace {int(low_pct*100)}-{int(high_pct*100)}% Completion",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Arial"
            )
        )
        
        figures.append(fig)
    
    return figures


def create_custom_percentile_frames(frame_data, coordinates_2d, labels_anchors, percentile_ranges=None, frames_per_percentile_range=1, connect_anchors=True, anchor_size=24, correct_path_color="green"):
    """
    Create frames based on custom percentile ranges of reasoning chain completion.
    Creates separate frames for correct and wrong chains to avoid overlap.
    
    Args:
        frame_data: Dictionary containing processed data for each frame
        coordinates_2d: 2D coordinates for plotting
        labels_anchors: List of anchor labels
        percentile_ranges: List of tuples with (start_pct, end_pct) for each frame
        frames_per_percentile_range: Number of frames to generate within each percentile range (default: 1)
            When > 1, each percentile range will be subdivided into smaller equal-sized chunks
        connect_anchors: Whether to connect anchor points with a line to show the correct path (default: True)
        anchor_size: Size of anchor markers (default: 24)
        correct_path_color: Color of the line connecting the correct path anchors (default: "green")
        
    Returns:
        List[go.Figure]: List of static frame figures with single contours
    """
    # Default to standard 5 percentile ranges if none provided
    if percentile_ranges is None:
        percentile_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    
    # List to collect all figures
    figures = []
    
    # Create a timestamp for unique identification
    timestamp = int(time.time())
    
    # Process each key in frame_data separately
    for frame_key, frame_item in frame_data.items():
        # Extract the necessary data for this frame
        num_thoughts_each_chain = frame_item["num_thoughts_each_chain"]
        num_chains = frame_item["num_chains"]
        all_answers = frame_item["all_answers"]
        answer_gt_short = frame_item["answer_gt_short"]
        
        # Process each percentile range
        for low_pct, high_pct in percentile_ranges:
            # Subdivide the current percentile range if requested
            if frames_per_percentile_range > 1:
                sub_ranges = []
                range_size = high_pct - low_pct
                sub_size = range_size / frames_per_percentile_range
                for i in range(frames_per_percentile_range):
                    sub_low = low_pct + (i * sub_size)
                    sub_high = low_pct + ((i + 1) * sub_size)
                    sub_ranges.append((sub_low, sub_high))
            else:
                sub_ranges = [(low_pct, high_pct)]
            
            # Process each sub-range (or just the original range if frames_per_percentile_range=1)
            for sub_low_pct, sub_high_pct in sub_ranges:
                # Create separate figures for wrong and correct chains
                wrong_fig = go.Figure()
                correct_fig = go.Figure()
                
                # FOR WRONG CHAINS FIGURE
                wrong_points = []
                for chain_idx in range(num_chains):
                    start_idx = sum(num_thoughts_each_chain[:chain_idx])
                    end_idx = sum(num_thoughts_each_chain[:chain_idx + 1])
                    
                    if end_idx <= start_idx:
                        continue
                    
                    # Calculate indices for this percentile range
                    chain_length = end_idx - start_idx
                    low_idx = start_idx + int(chain_length * sub_low_pct)
                    high_idx = start_idx + int(chain_length * sub_high_pct)
                    
                    # Only include points in this percentile range and only for wrong chains
                    if high_idx > low_idx and all_answers[chain_idx] != answer_gt_short:
                        chain_points = coordinates_2d[low_idx:high_idx]
                        wrong_points.append(chain_points)
                
                # Add contours for wrong chains (only for this percentile)
                if wrong_points and len(wrong_points) > 0:
                    try:
                        wrong_x = np.concatenate([p[:, 0] for p in wrong_points if len(p) > 0])
                        wrong_y = np.concatenate([p[:, 1] for p in wrong_points if len(p) > 0])
                        if len(wrong_x) > 0:
                            # Add ONLY wrong chains to the wrong figure
                            wrong_fig.add_trace(
                                go.Histogram2dContour(
                                    x=wrong_x,
                                    y=wrong_y,
                                    colorscale="Reds",
                                    showscale=False,
                                    histfunc='count',
                                    contours=dict(
                                        showlines=True,
                                        coloring='fill'
                                    ),
                                    autocontour=True,
                                    opacity=0.6,
                                    name=f'Wrong Chains {int(sub_low_pct*100)}-{int(sub_high_pct*100)}%'
                                )
                            )
                    except ValueError:
                        # Skip if there's an issue with concatenation
                        pass
                
                # FOR CORRECT CHAINS FIGURE
                correct_points = []
                for chain_idx in range(num_chains):
                    start_idx = sum(num_thoughts_each_chain[:chain_idx])
                    end_idx = sum(num_thoughts_each_chain[:chain_idx + 1])
                    
                    if end_idx <= start_idx:
                        continue
                    
                    # Calculate indices for this percentile range
                    chain_length = end_idx - start_idx
                    low_idx = start_idx + int(chain_length * sub_low_pct)
                    high_idx = start_idx + int(chain_length * sub_high_pct)
                    
                    # Only include points in this percentile range and only for correct chains
                    if high_idx > low_idx and all_answers[chain_idx] == answer_gt_short:
                        chain_points = coordinates_2d[low_idx:high_idx]
                        correct_points.append(chain_points)
                
                # Add contours for correct chains (only for this percentile)
                if correct_points and len(correct_points) > 0:
                    try:
                        correct_x = np.concatenate([p[:, 0] for p in correct_points if len(p) > 0])
                        correct_y = np.concatenate([p[:, 1] for p in correct_points if len(p) > 0])
                        if len(correct_x) > 0:
                            # Add ONLY correct chains to the correct figure
                            correct_fig.add_trace(
                                go.Histogram2dContour(
                                    x=correct_x,
                                    y=correct_y,
                                    colorscale="Blues",
                                    showscale=False,
                                    histfunc='count',
                                    contours=dict(
                                        showlines=True,
                                        coloring='fill'
                                    ),
                                    autocontour=True,
                                    opacity=0.6,
                                    name=f'Correct Chains {int(sub_low_pct*100)}-{int(sub_high_pct*100)}%'
                                )
                            )
                    except ValueError:
                        # Skip if there's an issue with concatenation
                        pass
                
                # Collect anchor coordinates for connecting line
                start_anchor_idx = None
                correct_anchor_idx = None
                anchor_x_coords = []
                anchor_y_coords = []
                anchor_names = []
                anchor_colors = []
                
                # Add anchor points to both figures
                for idx, anchor_name in enumerate(labels_anchors):
                    if idx >= len(coordinates_2d) - len(labels_anchors):
                        anchor_idx = len(coordinates_2d) - len(labels_anchors) + idx
                        
                        # Set marker style based on anchor type
                        if anchor_name == 'Start':
                            marker_symbol = 'star'
                            marker_color = 'blue'
                            start_anchor_idx = len(anchor_x_coords)
                        elif anchor_name == answer_gt_short:
                            marker_symbol = 'diamond'
                            marker_color = "green"
                            correct_anchor_idx = len(anchor_x_coords)
                        else:
                            marker_symbol = 'x'
                            marker_color = "red"
                        
                        if anchor_idx < len(coordinates_2d):
                            # Store coordinates for connecting line
                            anchor_x_coords.append(coordinates_2d[anchor_idx, 0])
                            anchor_y_coords.append(coordinates_2d[anchor_idx, 1])
                            anchor_names.append(anchor_name)
                            anchor_colors.append(marker_color)
                            
                            # Add markers to wrong chains figure
                            wrong_fig.add_trace(
                                go.Scatter(
                                    x=[coordinates_2d[anchor_idx, 0]],
                                    y=[coordinates_2d[anchor_idx, 1]],
                                    mode='markers+text',
                                    name=anchor_name,
                                    text=[anchor_name],
                                    textposition="top center",
                                    marker=dict(
                                        symbol=marker_symbol,
                                        size=anchor_size,
                                        line_width=2,
                                        color=marker_color
                                    ),
                                    hoverinfo='text',
                                    hovertext=anchor_name,
                                    showlegend=True
                                )
                            )
                            
                            # Add markers to correct chains figure
                            correct_fig.add_trace(
                                go.Scatter(
                                    x=[coordinates_2d[anchor_idx, 0]],
                                    y=[coordinates_2d[anchor_idx, 1]],
                                    mode='markers+text',
                                    name=anchor_name,
                                    text=[anchor_name],
                                    textposition="top center",
                                    marker=dict(
                                        symbol=marker_symbol,
                                        size=anchor_size,
                                        line_width=2,
                                        color=marker_color
                                    ),
                                    hoverinfo='text',
                                    hovertext=anchor_name,
                                    showlegend=True
                                )
                            )
                
                # Update layout for wrong chains figure - make it clearly ONLY wrong chains
                wrong_fig.update_layout(
                    xaxis=dict(
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
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray',
                        zeroline=False,
                        showline=True,
                        linewidth=1,
                        linecolor='black',
                        mirror=True,
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color="black"),
                    height=600,
                    width=900,
                    title=f"ONLY WRONG Chains [{frame_key}]: {int(sub_low_pct*100)}-{int(sub_high_pct*100)}% Completion",
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(255, 255, 255, 0.8)"
                    ),
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=16,
                        font_family="Arial"
                    )
                )
                
                # Update layout for correct chains figure - make it clearly ONLY correct chains
                correct_fig.update_layout(
                    xaxis=dict(
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
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray',
                        zeroline=False,
                        showline=True,
                        linewidth=1,
                        linecolor='black',
                        mirror=True,
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color="black"),
                    height=600,
                    width=900,
                    title=f"ONLY CORRECT Chains [{frame_key}]: {int(sub_low_pct*100)}-{int(sub_high_pct*100)}% Completion",
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(255, 255, 255, 0.8)"
                    ),
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=16,
                        font_family="Arial"
                    )
                )
                
                # Set metadata to help identify the figures
                wrong_fig.layout.meta = {
                    "type": "wrong", 
                    "timestamp": timestamp, 
                    "range": f"{int(sub_low_pct*100)}-{int(sub_high_pct*100)}",
                    "frame_key": frame_key
                }
                correct_fig.layout.meta = {
                    "type": "correct", 
                    "timestamp": timestamp, 
                    "range": f"{int(sub_low_pct*100)}-{int(sub_high_pct*100)}",
                    "frame_key": frame_key
                }
                
                # Add figures to the list - first wrong, then correct
                figures.append(wrong_fig)
                figures.append(correct_fig)
    
    return figures 