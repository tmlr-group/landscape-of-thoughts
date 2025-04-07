import plotly.graph_objects as go
import numpy as np

def create_snapshot_frames(frame_data, coordinates_2d, labels_anchors):
    """
    Create snapshot frames where each frame shows a different sample/method.
    
    Args:
        frame_data: Dictionary containing processed data for each frame
        coordinates_2d: 2D coordinates for plotting
        labels_anchors: List of anchor labels
        
    Returns:
        List[go.Figure]: List of static frame figures
    """
    figures = []
    
    # For each key in frame_data (each sample/method combination)
    for key, data in frame_data.items():
        fig = go.Figure()
        
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
            
            # Include all points for this chain
            chain_points = coordinates_2d[start_idx:end_idx]
            
            if all_answers[chain_idx] == answer_gt_short:
                correct_points.append(chain_points)
            else:
                wrong_points.append(chain_points)
    
        # Add contours for wrong chains
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
                            name='Wrong Chains'
                        )
                    )
            except ValueError:
                # Skip if there's an issue with concatenation
                pass
        
        # Add contours for correct chains
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
                            name='Correct Chains'
                        )
                    )
            except ValueError:
                # Skip if there's an issue with concatenation
                pass
    
        # Add anchor points
        for idx, anchor_name in enumerate(labels_anchors):
            if idx >= len(coordinates_2d) - len(labels_anchors):
                anchor_idx = idx - (len(coordinates_2d) - len(labels_anchors))
                if anchor_name == 'Start':
                    marker_symbol = 'star'
                    marker_color = 'blue'
                elif anchor_name == answer_gt_short:
                    marker_symbol = 'diamond'
                    marker_color = "green"
                else:
                    marker_symbol = 'x'
                    marker_color = "red"
                
                # Use the appropriate index to locate anchor points
                anchor_idx = len(coordinates_2d) - len(labels_anchors) + idx
                if anchor_idx < len(coordinates_2d):
                    fig.add_trace(
                        go.Scatter(
                            x=[coordinates_2d[anchor_idx, 0]],
                            y=[coordinates_2d[anchor_idx, 1]],
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
            title=f"Reasoning Trace Snapshot: {key}"
        )
        
        figures.append(fig)
    
    return figures 