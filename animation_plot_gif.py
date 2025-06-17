import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import gaussian_kde
from typing import Dict, List, Any, Tuple

from lot.visualization_utils import process_landscape_data
from lot.visualization_utils.utils import process_chain_points, split_list

def create_animation_from_chains(wrong_segments, correct_segments, label_anchor_points, 
                                label_anchor_symbols, label_anchor_colors, num_splits,
                                output_filename='chain_animation', speed_factor=0.8):
    """
    Create separate animations for wrong and correct chains
    
    Parameters:
    -----------
    wrong_segments : list of tuples
        List of (x, y) segments for wrong chains for each split
    correct_segments : list of tuples
        List of (x, y) segments for correct chains for each split
    label_anchor_points : numpy.ndarray
        Array of anchor points
    label_anchor_symbols : list
        List of symbols for anchor points
    label_anchor_colors : list
        List of colors for anchor points
    num_splits : int
        Number of splits in the animation
    output_filename : str
        Output filename for the animation
    speed_factor : float
        Speed factor for the animation
    """
    # Create separate figures for wrong and correct chains
    fig_wrong = plt.figure(figsize=(10, 8), dpi=200)
    ax1 = fig_wrong.add_subplot(111)
    
    fig_correct = plt.figure(figsize=(10, 8), dpi=200)
    ax2 = fig_correct.add_subplot(111)
    
    # Set titles
    ax1.set_title('Wrong Chains', fontsize=14)
    ax2.set_title('Correct Chains', fontsize=14)
    
    # Animation parameters
    frames_per_split = 20  # Number of frames per split
    transition_frames = 10  # Number of frames for transition between splits
    
    # Calculate the total number of frames
    total_frames = num_splits * frames_per_split + (num_splits - 1) * transition_frames
    
    # Find global min and max for consistent axis limits
    all_x_wrong = []
    all_y_wrong = []
    all_x_correct = []
    all_y_correct = []
    
    for i in range(num_splits):
        wrong_x, wrong_y = wrong_segments[i]
        correct_x, correct_y = correct_segments[i]
        
        all_x_wrong.extend(wrong_x)
        all_y_wrong.extend(wrong_y)
        all_x_correct.extend(correct_x)
        all_y_correct.extend(correct_y)
    
    all_x = all_x_wrong + all_x_correct + label_anchor_points[:, 0].tolist()
    all_y = all_y_wrong + all_y_correct + label_anchor_points[:, 1].tolist()
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # Add some padding
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    
    # Create grid for KDE
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    grid_points = np.vstack([X.ravel(), Y.ravel()])
    
    # Pre-calculate all KDEs for smoother transitions
    wrong_Z_values = []
    correct_Z_values = []
    
    for i in range(num_splits):
        # For wrong chains
        wrong_x_split, wrong_y_split = wrong_segments[i]
        if len(wrong_x_split) > 2:
            try:
                wrong_data = np.vstack([wrong_x_split, wrong_y_split])
                kde_wrong = gaussian_kde(wrong_data, bw_method='scott')
                Z_wrong = kde_wrong(grid_points).reshape(X.shape)
                wrong_Z_values.append(Z_wrong)
            except Exception as e:
                print(f"Warning: KDE calculation failed for wrong split {i}: {e}")
                wrong_Z_values.append(np.zeros_like(X))
        else:
            wrong_Z_values.append(np.zeros_like(X))

        # For correct chains
        correct_x_split, correct_y_split = correct_segments[i]
        if len(correct_x_split) > 2:
            try:
                correct_data = np.vstack([correct_x_split, correct_y_split])
                kde_correct = gaussian_kde(correct_data, bw_method='scott')
                Z_correct = kde_correct(grid_points).reshape(X.shape)
                correct_Z_values.append(Z_correct)
            except Exception as e:
                print(f"Warning: KDE calculation failed for correct split {i}: {e}")
                correct_Z_values.append(np.zeros_like(X))
        else:
            correct_Z_values.append(np.zeros_like(X))
    
    # Create separate update functions for wrong and correct chains
    def update_wrong(frame):
        # Calculate which split we're on and whether we're in a transition
        adjusted_frame = frame
        current_split_idx = 0
        in_transition = False
        transition_progress = 0
        
        for i in range(num_splits):
            if i > 0:
                if adjusted_frame < transition_frames:
                    current_split_idx = i - 1
                    in_transition = True
                    transition_progress = adjusted_frame / transition_frames
                    break
                adjusted_frame -= transition_frames
                
            if adjusted_frame < frames_per_split:
                current_split_idx = i
                in_transition = False
                break
            adjusted_frame -= frames_per_split
        
        current_split_idx = min(current_split_idx, num_splits - 1)

        # Check if there's data to plot for this frame
        should_update = False
        if not in_transition:
            # Update if the current split has data
            if len(wrong_segments[current_split_idx][0]) > 0:
                should_update = True
        else:
            # Update if either of the splits in transition has data
            next_split_idx = min(current_split_idx + 1, num_splits - 1)
            if len(wrong_segments[current_split_idx][0]) > 0 or len(wrong_segments[next_split_idx][0]) > 0:
                should_update = True
        
        if not should_update:
            return  # Skip updating this frame
            
        # Clear axes and reset properties
        ax1.clear()
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_title('Wrong Chains', fontsize=14)
        
        # Get current split data
        wrong_x, wrong_y = wrong_segments[current_split_idx]
        
        # Handle regular split display or transition
        if not in_transition:
            # Regular split display - show points progressively
            frame_within_split = adjusted_frame
            transition_factor = (frame_within_split + 1) / frames_per_split
            
            wrong_points_to_show = max(5, int(len(wrong_x) * transition_factor))
            
            wrong_x_partial = wrong_x[:wrong_points_to_show]
            wrong_y_partial = wrong_y[:wrong_points_to_show]
            
            ax1.scatter(wrong_x_partial, wrong_y_partial, c='red', s=10, alpha=0.9)
            
            # Use pre-calculated KDE and fade it in
            Z_wrong = wrong_Z_values[current_split_idx]
            ax1.contourf(X, Y, Z_wrong, levels=25, cmap='Reds', alpha=0.3)
            ax1.contour(X, Y, Z_wrong, levels=8, colors='darkred', alpha=0.201, linewidths=0.5)
            
            progress = int(transition_factor * 100)
            ax1.set_title(f'Wrong Chains - Split {current_split_idx+1}/{num_splits} ({progress}% complete)', fontsize=14)
            
        else:
            # Transition between splits
            next_split_idx = min(current_split_idx + 1, num_splits - 1)
            next_wrong_x, next_wrong_y = wrong_segments[next_split_idx]
            
            # Show points from the next split progressively
            wrong_points_to_show_next = max(5, int(len(next_wrong_x) * transition_progress))
            
            next_wrong_x_partial = next_wrong_x[:wrong_points_to_show_next]
            next_wrong_y_partial = next_wrong_y[:wrong_points_to_show_next]
            
            alpha_next = 0.9 * transition_progress
            ax1.scatter(next_wrong_x_partial, next_wrong_y_partial, c='red', s=10, alpha=alpha_next)
            
            # Blend pre-calculated KDEs
            Z_wrong_current = wrong_Z_values[current_split_idx]
            Z_wrong_next = wrong_Z_values[next_split_idx]
            Z_wrong_blend = (1 - transition_progress) * Z_wrong_current + transition_progress * Z_wrong_next

            ax1.contourf(X, Y, Z_wrong_blend, levels=25, cmap='Reds', alpha=0.3)
            ax1.contour(X, Y, Z_wrong_blend, levels=8, colors='darkred', alpha=0.2, linewidths=0.5)

            ax1.set_title(f'Wrong Chains - Transitioning {current_split_idx+1} → {next_split_idx+1}', fontsize=14)
        
        # Always plot anchor points
        for i in range(len(label_anchor_points)):
            ax1.scatter(
                label_anchor_points[i, 0], label_anchor_points[i, 1], 
                c=label_anchor_colors[i], s=200, alpha=0.9, marker=label_anchor_symbols[i],
                edgecolors='black', linewidths=1.0
            )
        
        fig_wrong.tight_layout()
    
    def update_correct(frame):
        # Calculate which split we're on and whether we're in a transition
        adjusted_frame = frame
        current_split_idx = 0
        in_transition = False
        transition_progress = 0
        
        for i in range(num_splits):
            if i > 0:
                if adjusted_frame < transition_frames:
                    current_split_idx = i - 1
                    in_transition = True
                    transition_progress = adjusted_frame / transition_frames
                    break
                adjusted_frame -= transition_frames
                
            if adjusted_frame < frames_per_split:
                current_split_idx = i
                in_transition = False
                break
            adjusted_frame -= frames_per_split
        
        current_split_idx = min(current_split_idx, num_splits - 1)

        # Check if there's data to plot for this frame
        should_update = False
        if not in_transition:
            # Update if the current split has data
            if len(correct_segments[current_split_idx][0]) > 0:
                should_update = True
        else:
            # Update if either of the splits in transition has data
            next_split_idx = min(current_split_idx + 1, num_splits - 1)
            if len(correct_segments[current_split_idx][0]) > 0 or len(correct_segments[next_split_idx][0]) > 0:
                should_update = True
        
        if not should_update:
            return  # Skip updating this frame
            
        # Clear axes and reset properties
        ax2.clear()
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax2.set_title('Correct Chains', fontsize=14)
        
        # Get current split data
        correct_x, correct_y = correct_segments[current_split_idx]
        
        # Handle regular split display or transition
        if not in_transition:
            # Regular split display - show points progressively
            frame_within_split = adjusted_frame
            transition_factor = (frame_within_split + 1) / frames_per_split
            
            correct_points_to_show = max(5, int(len(correct_x) * transition_factor))
            
            correct_x_partial = correct_x[:correct_points_to_show]
            correct_y_partial = correct_y[:correct_points_to_show]
            
            ax2.scatter(correct_x_partial, correct_y_partial, c='blue', s=10, alpha=0.9)
            
            # Use pre-calculated KDE and fade it in
            Z_correct = correct_Z_values[current_split_idx]
            ax2.contourf(X, Y, Z_correct, levels=25, cmap='Blues', alpha=0.3)
            ax2.contour(X, Y, Z_correct, levels=8, colors='darkblue', alpha=0.201, linewidths=0.5)

            progress = int(transition_factor * 100)
            ax2.set_title(f'Correct Chains - Split {current_split_idx+1}/{num_splits} ({progress}% complete)', fontsize=14)
            
        else:
            # Transition between splits
            next_split_idx = min(current_split_idx + 1, num_splits - 1)
            next_correct_x, next_correct_y = correct_segments[next_split_idx]
            
            # Show points from the next split progressively
            correct_points_to_show_next = max(5, int(len(next_correct_x) * transition_progress))
            
            next_correct_x_partial = next_correct_x[:correct_points_to_show_next]
            next_correct_y_partial = next_correct_y[:correct_points_to_show_next]
            
            alpha_next = 0.9 * transition_progress
            ax2.scatter(next_correct_x_partial, next_correct_y_partial, c='blue', s=10, alpha=alpha_next)
            
            # Blend pre-calculated KDEs
            Z_correct_current = correct_Z_values[current_split_idx]
            Z_correct_next = correct_Z_values[next_split_idx]
            Z_correct_blend = (1 - transition_progress) * Z_correct_current + transition_progress * Z_correct_next
            
            ax2.contourf(X, Y, Z_correct_blend, levels=25, cmap='Blues', alpha=0.3)
            ax2.contour(X, Y, Z_correct_blend, levels=8, colors='darkblue', alpha=0.2, linewidths=0.5)
            
            ax2.set_title(f'Correct Chains - Transitioning {current_split_idx+1} → {next_split_idx+1}', fontsize=14)
        
        # Always plot anchor points
        for i in range(len(label_anchor_points)):
            ax2.scatter(
                label_anchor_points[i, 0], label_anchor_points[i, 1], 
                c=label_anchor_colors[i], s=200, alpha=0.9, marker=label_anchor_symbols[i],
                edgecolors='black', linewidths=1.0
            )
        
        fig_correct.tight_layout()
    
    # Create animations
    ani_wrong = FuncAnimation(fig_wrong, update_wrong, frames=total_frames, blit=False)
    ani_correct = FuncAnimation(fig_correct, update_correct, frames=total_frames, blit=False)
    
    # Calculate fps based on speed factor
    base_fps = 20
    adjusted_fps = int(base_fps * speed_factor)
    
    # Save the animations
    ani_wrong.save(output_filename + '_wrong.gif', writer='pillow', fps=adjusted_fps, dpi=200)
    ani_correct.save(output_filename + '_correct.gif', writer='pillow', fps=adjusted_fps, dpi=200)
    
    # Save as mp4 if ffmpeg is available
    ani_wrong.save(output_filename + '_wrong.mp4', writer='ffmpeg', fps=adjusted_fps, dpi=200)
    ani_correct.save(output_filename + '_correct.mp4', writer='ffmpeg', fps=adjusted_fps, dpi=200)
    
    # Return the animation objects
    return ani_wrong, ani_correct

def get_datapoints(
    dataset_name: str, 
    plot_datas: Dict[int, Dict[str, Any]], 
    splited_T_2D: List[np.ndarray], 
    A_matrix_2D: np.ndarray, 
    num_all_thoughts_w_start_list: List[int],
    num_splits: int = 5
) -> List[Tuple[np.ndarray, str, str]]:
    all_T_with_start_coordinate_matrix = split_list(num_all_thoughts_w_start_list, splited_T_2D)

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

    # Calculate thresholds dynamically based on number of subfigures
    percentiles = [i * (100 / num_splits) for i in range(1, num_splits)]
    default_thresholds = np.array([i / num_splits for i in range(1, num_splits)])
    
    # Calculate thresholds for both sets
    wrong_thresholds = np.percentile(wrong_weights, percentiles) if len(wrong_weights) > 0 else default_thresholds
    
    # Handle the case where there are no correct answers
    if len(correct_weights) > 0:
        correct_thresholds = np.percentile(correct_weights, percentiles)
    else:
        print("Warning: No correct answers found. Using default thresholds for correct answers.")
        correct_thresholds = default_thresholds
        # Create empty arrays for correct segments
        correct_x = np.array([])
        correct_y = np.array([])

    # Lists to store segment data
    wrong_segments = []
    correct_segments = []
    # Process wrong chains
    for i in range(num_splits):
        if i == 0:
            wrong_mask = wrong_weights <= wrong_thresholds[0] if len(wrong_weights) > 0 else np.array([], dtype=bool)
            correct_mask = correct_weights <= correct_thresholds[0] if len(correct_weights) > 0 else np.array([], dtype=bool)
        elif i == num_splits - 1:
            wrong_mask = wrong_weights > wrong_thresholds[-1] if len(wrong_weights) > 0 else np.array([], dtype=bool)
            correct_mask = correct_weights > correct_thresholds[-1] if len(correct_weights) > 0 else np.array([], dtype=bool)
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

    # Add anchors to both plots
    if dataset_name == "mmlu":
        labels_anchors = ['A', 'B', 'C', 'D']
    elif dataset_name == "strategyqa":
        labels_anchors = ['A', 'B']
    else:
        labels_anchors = ['A', 'B', 'C', 'D', 'E']

    label_anchor_points = []
    label_anchor_symbols = []
    label_anchor_colors = []
    # Add anchors to both subplots
    for idx, anchor_name in enumerate(labels_anchors):
        if idx == 0:  # the first anchor is the correct one 
            marker_symbol = '*'
            marker_color = "green"
        else: 
            marker_symbol = 'X'
            marker_color = "red"

        label_anchor_points.append(A_matrix_2D[idx, :])
        label_anchor_symbols.append(marker_symbol)
        label_anchor_colors.append(marker_color)
    
    return label_anchor_points, label_anchor_symbols, label_anchor_colors, wrong_segments, correct_segments


plt.style.use('default')

import time
t0 = time.time()

# Generate and display animated plots
method_idx = 0
num_splits = 15

model_name: str = 'Meta-Llama-3.1-70B-Instruct-Turbo'
dataset_name: str = 'aqua'
method: str = 'cot'
plot_type: str = 'method'
save_root: str = "Landscape-Data"
output_dir: str = "figures/landscape"

print(f"==> model_name: {model_name}")
print(f"==> dataset_name: {dataset_name}")
print(f"==> method: {method}")
print(f"==> plot_type: {plot_type}")
print(f"==> save_root: {save_root}")
print(f"==> output_dir: {output_dir}")

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

print("==> Creating animations...")
for plot_datas, splited_T_2D, num_all_thoughts_w_start_list in zip(list_plot_data, list_all_T_2D, list_num_all_thoughts_w_start_list):
    # Create the landscape figure with customizable number of subfigures
    label_anchor_points, label_anchor_symbols, label_anchor_colors, wrong_segments, correct_segments = get_datapoints(
        dataset_name=dataset_name,
        plot_datas=plot_datas,
        splited_T_2D=splited_T_2D,
        A_matrix_2D=A_matrix_2D,
        num_all_thoughts_w_start_list=num_all_thoughts_w_start_list,
        num_splits=num_splits  # You can change this to any number of subfigures you want (e.g., 3, 7, 10)
    )

    # (array([20.187387, 70.699524], dtype=float32), 'star', 'green')
    label_anchor_points = np.stack(label_anchor_points, axis=0)
    
    # Create separate animations for wrong and correct chains
    ani_wrong, ani_correct = create_animation_from_chains(
        wrong_segments=wrong_segments,
        correct_segments=correct_segments,
        label_anchor_points=label_anchor_points,
        label_anchor_symbols=label_anchor_symbols,
        label_anchor_colors=label_anchor_colors,
        num_splits=num_splits,
        output_filename=f'chain_animation'
    )
    
    print(f"Created animations at 'chain_animation_wrong.gif' and 'chain_animation_correct.gif'")
    print(f"MP4 versions may also be available if ffmpeg is installed")

print(f"==> Time taken: {time.time() - t0} seconds")