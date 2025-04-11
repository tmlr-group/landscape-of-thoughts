import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Any

from .visualization_utils.utils import process_chain_points, split_list
from .visualization_utils.landscape import process_landscape_data
import plotly.io as pio
import os
from fire import Fire

def create_animations(
    dataset_name: str, 
    plot_datas: Dict[int, Dict[str, Any]], 
    splited_T_2D: List[np.ndarray], 
    A_matrix_2D: np.ndarray, 
    num_all_thoughts_w_start_list: List[int],
    num_frames: int = 5
) -> tuple:
    """
    Create two separate animated landscape visualizations (correct and wrong chains).
    
    Args:
        dataset_name (str): Name of the dataset.
        plot_datas (Dict[int, Dict[str, Any]]): Data for plotting.
        splited_T_2D (List[np.ndarray]): Split T matrix in 2D.
        A_matrix_2D (np.ndarray): A matrix in 2D.
        num_all_thoughts_w_start_list (List[int]): List of number of thoughts with start.
        num_frames (int, optional): Number of frames to display. Defaults to 5.
        
    Returns:
        tuple: Two Plotly figure objects (wrong_fig, correct_fig).
    """
    all_T_with_start_coordinate_matrix = split_list(num_all_thoughts_w_start_list, splited_T_2D)

    # Collect points and separate them for correct/wrong chains
    wrong_chain_points = []
    correct_chain_points = []
    all_start_coordinates = []
    answer_gt_short = None  # Will be set in the loop
    
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

    # Process both chains
    wrong_x, wrong_y, wrong_weights, _, _ = process_chain_points(wrong_chain_points)
    correct_x, correct_y, correct_weights, _, _ = process_chain_points(correct_chain_points)

    # Calculate thresholds for both sets
    percentiles = [100 * (i+1) / num_frames for i in range(num_frames-1)]
    wrong_thresholds = np.percentile(wrong_weights, percentiles) if len(wrong_weights) > 0 else np.array([0.2, 0.4, 0.6, 0.8])[:num_frames-1]
    
    if len(correct_weights) > 0:
        correct_thresholds = np.percentile(correct_weights, percentiles)
    else:
        print("Warning: No correct answers found. Using default thresholds for correct answers.")
        correct_thresholds = np.array([0.2, 0.4, 0.6, 0.8])[:num_frames-1]
        correct_x = np.array([])
        correct_y = np.array([])

    # Get anchor labels based on dataset
    if dataset_name == "mmlu":
        labels_anchors = ['A', 'B', 'C', 'D']
    elif dataset_name == "strategyqa":
        labels_anchors = ['A', 'B']
    else:
        labels_anchors = ['A', 'B', 'C', 'D', 'E']

    # Generate frame labels
    percentile_ranges = np.linspace(0, 100, num_frames+1).astype(int)
    frame_labels = [f'{percentile_ranges[i]}-{percentile_ranges[i+1]}%' for i in range(num_frames)]

    # Process data into segments for animation frames
    wrong_segments = []
    correct_segments = []
    
    # Generate blended segments for smoother transitions
    num_blend_frames = 5  # Number of intermediate frames for smooth transition
    
    # Create more fine-grained thresholds for smoother transitions
    total_blend_frames = num_frames * num_blend_frames
    blend_percentiles = [100 * (i+1) / total_blend_frames for i in range(total_blend_frames-1)]
    
    # Calculate fine-grained thresholds
    if len(wrong_weights) > 0:
        wrong_blend_thresholds = np.percentile(wrong_weights, blend_percentiles)
    else:
        wrong_blend_thresholds = np.linspace(0.2, 0.8, total_blend_frames-1)
        
    if len(correct_weights) > 0:
        correct_blend_thresholds = np.percentile(correct_weights, blend_percentiles)
    else:
        correct_blend_thresholds = np.linspace(0.2, 0.8, total_blend_frames-1)
    
    # Generate segments for wrong chains
    for i in range(total_blend_frames):
        if i == 0:
            # First segment
            wrong_mask = wrong_weights <= wrong_blend_thresholds[0] if len(wrong_weights) > 0 else np.array([], dtype=bool)
        elif i == total_blend_frames-1:
            # Last segment
            wrong_mask = wrong_weights > wrong_blend_thresholds[-1] if len(wrong_weights) > 0 else np.array([], dtype=bool)
        else:
            # Middle segments
            if len(wrong_weights) > 0:
                wrong_mask = (wrong_weights > wrong_blend_thresholds[i-1]) & (wrong_weights <= wrong_blend_thresholds[i])
            else:
                wrong_mask = np.array([], dtype=bool)
        
        # Get segments
        if len(wrong_weights) > 0:
            wrong_x_segment = np.array(wrong_x)[wrong_mask]
            wrong_y_segment = np.array(wrong_y)[wrong_mask]
        else:
            wrong_x_segment = np.array([])
            wrong_y_segment = np.array([])
            
        # Store segments
        wrong_segments.append((wrong_x_segment, wrong_y_segment))
    
    # Generate segments for correct chains
    for i in range(total_blend_frames):
        if i == 0:
            # First segment
            correct_mask = correct_weights <= correct_blend_thresholds[0] if len(correct_weights) > 0 else np.array([], dtype=bool)
        elif i == total_blend_frames-1:
            # Last segment
            correct_mask = correct_weights > correct_blend_thresholds[-1] if len(correct_weights) > 0 else np.array([], dtype=bool)
        else:
            # Middle segments
            if len(correct_weights) > 0:
                correct_mask = (correct_weights > correct_blend_thresholds[i-1]) & (correct_weights <= correct_blend_thresholds[i])
            else:
                correct_mask = np.array([], dtype=bool)
        
        # Get segments
        if len(correct_weights) > 0:
            correct_x_segment = np.array(correct_x)[correct_mask]
            correct_y_segment = np.array(correct_y)[correct_mask]
        else:
            correct_x_segment = np.array([])
            correct_y_segment = np.array([])
            
        # Store segments
        correct_segments.append((correct_x_segment, correct_y_segment))
    
    # Create WRONG chains animation
    #################################
    wrong_fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }
    
    # Setup wrong layout
    wrong_fig_dict["layout"] = {
        "title": f"Wrong Chains Landscape ({dataset_name.upper()})",
        "width": 800,
        "height": 600,
        "hovermode": "closest",
        "xaxis": {"title": "Dimension 1"},
        "yaxis": {"title": "Dimension 2"},
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "updatemenus": [{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 300, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 150,
                                                                     "easing": "cubic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate",
                                     "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    }
    
    # Create sliders for wrong animation
    wrong_sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 16},
            "prefix": "Percentile Range: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 150, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }
    
    # Add initial traces (first frame data)
    wrong_x_segment, wrong_y_segment = wrong_segments[0]
    
    # Add base traces for wrong chains
    if len(wrong_x_segment) > 0:
        wrong_fig_dict["data"].append({
            "type": "histogram2dcontour",
            "x": wrong_x_segment,
            "y": wrong_y_segment,
            "colorscale": "Reds",
            "showscale": True,
            "colorbar": {"title": "Density"},
            "histfunc": "count",
            "contours": {
                "showlines": True,
                "coloring": "fill"
            },
            "autocontour": True,
            "opacity": 0.7,
            "name": "Wrong Chains"
        })
    else:
        # Add an empty trace
        wrong_fig_dict["data"].append({
            "type": "scatter",
            "x": [],
            "y": [],
            "mode": "markers",
            "showlegend": False,
            "name": "Wrong Chains"
        })
    
    # Add anchors to wrong animation
    for idx, anchor_name in enumerate(labels_anchors):
        marker_symbol = 'star' if idx == 0 else 'x'  # First anchor is correct
        marker_color = "green" if idx == 0 else "red"
        
        wrong_fig_dict["data"].append({
            "type": "scatter",
            "x": [A_matrix_2D[idx, 0]],
            "y": [A_matrix_2D[idx, 1]],
            "mode": "markers+text",
            "text": anchor_name,
            "textposition": "top center",
            "marker": {
                "symbol": marker_symbol,
                "size": 18,
                "line": {"width": 1},
                "color": marker_color,
                "opacity": 0.9,
            },
            "name": f"{anchor_name} ({('Correct' if idx == 0 else 'Wrong')})"
        })
    
    # Create frames for wrong animation
    for i in range(0, total_blend_frames, num_blend_frames):
        frame_index = i // num_blend_frames
        frame = {"data": [], "name": frame_labels[frame_index]}
        wrong_x_segment, wrong_y_segment = wrong_segments[i]
        
        # Add wrong chains
        if len(wrong_x_segment) > 0:
            frame["data"].append({
                "type": "histogram2dcontour",
                "x": wrong_x_segment,
                "y": wrong_y_segment,
                "colorscale": "Reds",
                "showscale": True,
                "colorbar": {"title": "Density"},
                "histfunc": "count",
                "contours": {
                    "showlines": True,
                    "coloring": "fill"
                },
                "autocontour": True,
                "opacity": 0.7,
                "name": "Wrong Chains"
            })
        else:
            # Add an empty trace
            frame["data"].append({
                "type": "scatter",
                "x": [],
                "y": [],
                "mode": "markers",
                "showlegend": False,
                "name": "Wrong Chains"
            })
        
        # Add anchors to each frame
        for idx, anchor_name in enumerate(labels_anchors):
            marker_symbol = 'star' if idx == 0 else 'x'  # First anchor is correct
            marker_color = "green" if idx == 0 else "red"
            
            frame["data"].append({
                "type": "scatter",
                "x": [A_matrix_2D[idx, 0]],
                "y": [A_matrix_2D[idx, 1]],
                "mode": "markers+text",
                "text": anchor_name,
                "textposition": "top center",
                "marker": {
                    "symbol": marker_symbol,
                    "size": 18,
                    "line": {"width": 1},
                    "color": marker_color,
                    "opacity": 0.9,
                },
                "showlegend": False,
                "name": f"{anchor_name} ({('Correct' if idx == 0 else 'Wrong')})"
            })
        
        wrong_fig_dict["frames"].append(frame)
        
        # Add slider steps for main frames
        slider_step = {
            "args": [
                [frame_labels[frame_index]],
                {"frame": {"duration": 300, "redraw": True},
                "mode": "immediate",
                "transition": {"duration": 150}}
            ],
            "label": frame_labels[frame_index],
            "method": "animate"
        }
        wrong_sliders_dict["steps"].append(slider_step)
    
        # Add blend frames for smooth transition
        if frame_index < num_frames - 1:
            for blend_i in range(1, num_blend_frames):
                blend_idx = i + blend_i
                blend_frame = {"data": [], "name": f"blend_{frame_index}_{blend_i}"}
                blend_x_segment, blend_y_segment = wrong_segments[blend_idx]
                
                # Add wrong chains for blend frame
                if len(blend_x_segment) > 0:
                    blend_frame["data"].append({
                        "type": "histogram2dcontour",
                        "x": blend_x_segment,
                        "y": blend_y_segment,
                        "colorscale": "Reds",
                        "showscale": True,
                        "colorbar": {"title": "Density"},
                        "histfunc": "count",
                        "contours": {
                            "showlines": True,
                            "coloring": "fill"
                        },
                        "autocontour": True,
                        "opacity": 0.7,
                        "name": "Wrong Chains"
                    })
                else:
                    # Add an empty trace
                    blend_frame["data"].append({
                        "type": "scatter",
                        "x": [],
                        "y": [],
                        "mode": "markers",
                        "showlegend": False,
                        "name": "Wrong Chains"
                    })
                
                # Add anchors to blend frame
                for idx, anchor_name in enumerate(labels_anchors):
                    marker_symbol = 'star' if idx == 0 else 'x'  # First anchor is correct
                    marker_color = "green" if idx == 0 else "red"
                    
                    blend_frame["data"].append({
                        "type": "scatter",
                        "x": [A_matrix_2D[idx, 0]],
                        "y": [A_matrix_2D[idx, 1]],
                        "mode": "markers+text",
                        "text": anchor_name,
                        "textposition": "top center",
                        "marker": {
                            "symbol": marker_symbol,
                            "size": 18,
                            "line": {"width": 1},
                            "color": marker_color,
                            "opacity": 0.9,
                        },
                        "showlegend": False,
                        "name": f"{anchor_name} ({('Correct' if idx == 0 else 'Wrong')})"
                    })
                
                wrong_fig_dict["frames"].append(blend_frame)
    
    # Add slider to layout
    wrong_fig_dict["layout"]["sliders"] = [wrong_sliders_dict]
    
    # Create the wrong figure
    wrong_fig = go.Figure(wrong_fig_dict)
    
    # Create CORRECT chains animation
    ###################################
    correct_fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }
    
    # Setup correct layout
    correct_fig_dict["layout"] = {
        "title": f"Correct Chains Landscape ({dataset_name.upper()})",
        "width": 800,
        "height": 600,
        "hovermode": "closest",
        "xaxis": {"title": "Dimension 1"},
        "yaxis": {"title": "Dimension 2"},
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "updatemenus": [{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 300, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 150,
                                                                     "easing": "cubic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate",
                                     "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    }
    
    # Create sliders for correct animation
    correct_sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 16},
            "prefix": "Percentile Range: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 150, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }
    
    # Add initial traces (first frame data)
    correct_x_segment, correct_y_segment = correct_segments[0]
    
    # Add base traces for correct chains
    if len(correct_x_segment) > 0:
        correct_fig_dict["data"].append({
            "type": "histogram2dcontour",
            "x": correct_x_segment,
            "y": correct_y_segment,
            "colorscale": "Blues",
            "showscale": True,
            "colorbar": {"title": "Density"},
            "histfunc": "count",
            "contours": {
                "showlines": True,
                "coloring": "fill"
            },
            "autocontour": True,
            "opacity": 0.7,
            "name": "Correct Chains"
        })
    else:
        # Add an empty trace
        correct_fig_dict["data"].append({
            "type": "scatter",
            "x": [],
            "y": [],
            "mode": "markers",
            "showlegend": False,
            "name": "Correct Chains"
        })
    
    # Add anchors to correct animation
    for idx, anchor_name in enumerate(labels_anchors):
        marker_symbol = 'star' if idx == 0 else 'x'  # First anchor is correct
        marker_color = "green" if idx == 0 else "red"
        
        correct_fig_dict["data"].append({
            "type": "scatter",
            "x": [A_matrix_2D[idx, 0]],
            "y": [A_matrix_2D[idx, 1]],
            "mode": "markers+text",
            "text": anchor_name,
            "textposition": "top center",
            "marker": {
                "symbol": marker_symbol,
                "size": 18,
                "line": {"width": 1},
                "color": marker_color,
                "opacity": 0.9,
            },
            "name": f"{anchor_name} ({('Correct' if idx == 0 else 'Wrong')})"
        })
    
    # Create frames for correct animation with blending
    for i in range(0, total_blend_frames, num_blend_frames):
        frame_index = i // num_blend_frames
        frame = {"data": [], "name": frame_labels[frame_index]}
        correct_x_segment, correct_y_segment = correct_segments[i]
        
        # Add correct chains
        if len(correct_x_segment) > 0:
            frame["data"].append({
                "type": "histogram2dcontour",
                "x": correct_x_segment,
                "y": correct_y_segment,
                "colorscale": "Blues",
                "showscale": True,
                "colorbar": {"title": "Density"},
                "histfunc": "count",
                "contours": {
                    "showlines": True,
                    "coloring": "fill"
                },
                "autocontour": True,
                "opacity": 0.7,
                "name": "Correct Chains"
            })
        else:
            # Add an empty trace
            frame["data"].append({
                "type": "scatter",
                "x": [],
                "y": [],
                "mode": "markers",
                "showlegend": False,
                "name": "Correct Chains"
            })
        
        # Add anchors to each frame
        for idx, anchor_name in enumerate(labels_anchors):
            marker_symbol = 'star' if idx == 0 else 'x'  # First anchor is correct
            marker_color = "green" if idx == 0 else "red"
            
            frame["data"].append({
                "type": "scatter",
                "x": [A_matrix_2D[idx, 0]],
                "y": [A_matrix_2D[idx, 1]],
                "mode": "markers+text",
                "text": anchor_name,
                "textposition": "top center",
                "marker": {
                    "symbol": marker_symbol,
                    "size": 18,
                    "line": {"width": 1},
                    "color": marker_color,
                    "opacity": 0.9,
                },
                "showlegend": False,
                "name": f"{anchor_name} ({('Correct' if idx == 0 else 'Wrong')})"
            })
        
        correct_fig_dict["frames"].append(frame)
        
        # Add slider steps for main frames
        slider_step = {
            "args": [
                [frame_labels[frame_index]],
                {"frame": {"duration": 300, "redraw": True},
                "mode": "immediate",
                "transition": {"duration": 150}}
            ],
            "label": frame_labels[frame_index],
            "method": "animate"
        }
        correct_sliders_dict["steps"].append(slider_step)
        
        # Add blend frames for smooth transition
        if frame_index < num_frames - 1:
            for blend_i in range(1, num_blend_frames):
                blend_idx = i + blend_i
                blend_frame = {"data": [], "name": f"blend_{frame_index}_{blend_i}"}
                blend_x_segment, blend_y_segment = correct_segments[blend_idx]
                
                # Add correct chains for blend frame
                if len(blend_x_segment) > 0:
                    blend_frame["data"].append({
                        "type": "histogram2dcontour",
                        "x": blend_x_segment,
                        "y": blend_y_segment,
                        "colorscale": "Blues",
                        "showscale": True,
                        "colorbar": {"title": "Density"},
                        "histfunc": "count",
                        "contours": {
                            "showlines": True,
                            "coloring": "fill"
                        },
                        "autocontour": True,
                        "opacity": 0.7,
                        "name": "Correct Chains"
                    })
                else:
                    # Add an empty trace
                    blend_frame["data"].append({
                        "type": "scatter",
                        "x": [],
                        "y": [],
                        "mode": "markers",
                        "showlegend": False,
                        "name": "Correct Chains"
                    })
                
                # Add anchors to blend frame
                for idx, anchor_name in enumerate(labels_anchors):
                    marker_symbol = 'star' if idx == 0 else 'x'  # First anchor is correct
                    marker_color = "green" if idx == 0 else "red"
                    
                    blend_frame["data"].append({
                        "type": "scatter",
                        "x": [A_matrix_2D[idx, 0]],
                        "y": [A_matrix_2D[idx, 1]],
                        "mode": "markers+text",
                        "text": anchor_name,
                        "textposition": "top center",
                        "marker": {
                            "symbol": marker_symbol,
                            "size": 18,
                            "line": {"width": 1},
                            "color": marker_color,
                            "opacity": 0.9,
                        },
                        "showlegend": False,
                        "name": f"{anchor_name} ({('Correct' if idx == 0 else 'Wrong')})"
                    })
                
                correct_fig_dict["frames"].append(blend_frame)
    
    # Add slider to layout
    correct_fig_dict["layout"]["sliders"] = [correct_sliders_dict]
    
    # Create the correct figure
    correct_fig = go.Figure(correct_fig_dict)
    
    return wrong_fig, correct_fig


def main(
    model_name: str = 'Meta-Llama-3.1-8B-Instruct-Turbo',
    dataset_name: str = 'aqua',
    method: str = '',
    plot_type: str = 'method',
    num_frames: int = 10,
    save_root: str = "Landscape-Data",
    output_dir: str = "figures/animation"
) -> bool:
    # Create methods list
    methods = [method] if method else ['cot', 'l2m', 'mcts', 'tot']

    # Process data for landscape visualization
    list_all_T_2D, A_matrix_2D, list_plot_data, list_num_all_thoughts_w_start_list = process_landscape_data(
        model=model_name,
        dataset=dataset_name,
        methods=methods,
        plot_type=plot_type,
        ROOT=save_root
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save plots
    method_idx = 0
    for plot_datas, splited_T_2D, num_all_thoughts_w_start_list in zip(list_plot_data, list_all_T_2D, list_num_all_thoughts_w_start_list):
        # Create separate animated figures for wrong and correct chains
        wrong_fig, correct_fig = create_animations(
            dataset_name=dataset_name,
            plot_datas=plot_datas,
            splited_T_2D=splited_T_2D,
            A_matrix_2D=A_matrix_2D,
            num_all_thoughts_w_start_list=num_all_thoughts_w_start_list,
            num_frames=num_frames
        )
        
        save_path = os.path.join(output_dir, f"Correct-Animation-{model_name}-{dataset_name}-{methods[method_idx]}.html")
        print(f"==> Saving correct figure to: {save_path}")
        pio.write_html(correct_fig, save_path)
        
        save_path = os.path.join(output_dir, f"Wrong-Animation-{model_name}-{dataset_name}-{methods[method_idx]}.html")
        print(f"==> Saving wrong figure to: {save_path}")
        pio.write_html(wrong_fig, save_path)
        
        # Increment method index if not specific method
        if not method:
            method_idx += 1
    
if __name__ == "__main__":
    Fire(main)