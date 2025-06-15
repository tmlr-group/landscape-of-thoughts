import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
from plotly.subplots import make_subplots
from .visualization_utils.utils import process_chain_points, split_list
from .visualization_utils.landscape import process_landscape_data
import plotly.io as pio
import os
from fire import Fire


def get_points_wrong_and_correct_chains(
    plot_datas: Dict[int, Dict[str, Any]], 
    splited_T_2D: List[np.ndarray], 
    num_all_thoughts_w_start_list: List[int],
) -> tuple:
    """
    Create two separate animated landscape visualizations (correct and wrong chains).
    
    Args:
        plot_datas (Dict[int, Dict[str, Any]]): Data for plotting.
        splited_T_2D (List[np.ndarray]): Split T matrix in 2D.
        num_all_thoughts_w_start_list (List[int]): List of number of thoughts with start.
        
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
    wrong_x, wrong_y, _, _, _ = process_chain_points(wrong_chain_points)
    correct_x, correct_y, _, _, _ = process_chain_points(correct_chain_points)

    return wrong_x, wrong_y, correct_x, correct_y



class PlotlySlidingContourVisualizer:
    def __init__(self, x_data, y_data, window_size=100, step_size=10, color_theme='blue', 
                 fixed_points_matrix=None, point_symbols=None, point_colors=None, 
                 plot_width=900, plot_height=700):
        """
        Initialize the Plotly sliding window contour visualizer.
        
        Parameters:
        -----------
        x_data, y_data : array-like
            Lists or arrays of x and y coordinates.
        window_size : int
            Number of points to include in each window.
        step_size : int
            Number of points to advance the window in each frame.
        color_theme : str
            Color theme for the contour plot. Options: 'blue', 'red', 'green', 'viridis'
        fixed_points_matrix : ndarray, optional
            2D matrix where each row contains [x, y] coordinates of a fixed point (like A_matrix_2D)
        point_symbols : list, optional
            List of symbols for each point in fixed_points_matrix
        point_colors : list, optional
            List of colors for each point in fixed_points_matrix
        plot_width : int
            Width of the plot in pixels
        plot_height : int
            Height of the plot in pixels
        """
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.window_size = min(window_size, len(x_data))
        self.step_size = step_size
        self.color_theme = color_theme
        self.fixed_points_matrix = fixed_points_matrix
        self.plot_width = plot_width
        self.plot_height = plot_height
        
        # Default symbols and colors for matrix points if not provided
        self.point_symbols = point_symbols if point_symbols is not None else ['circle'] * (len(fixed_points_matrix) if fixed_points_matrix is not None else 0)
        self.point_colors = point_colors if point_colors is not None else ['black'] * (len(fixed_points_matrix) if fixed_points_matrix is not None else 0)
        
        
        # Set colorscale based on color_theme
        if color_theme == 'blue':
            color_list = px.colors.sequential.Blues
            self.colorscale = [[0, 'rgba(255,255,255,0)'], [0.1, color_list[1]], 
                    [0.5, color_list[3]], [1, color_list[5]]]
            self.scatter_color = color_list[5]  # Use a darker blue for scatter points
        elif color_theme == 'red':
            # Override the first color to be transparent white for lowest density
            color_list = px.colors.sequential.Reds
            self.colorscale = [[0, 'rgba(255,255,255,0)'], [0.1, color_list[1]], 
                                [0.5, color_list[3]], [1, color_list[5]]]
            self.scatter_color = color_list[5]  # Use a darker red for scatter points
        elif color_theme == 'green':
            color_list = px.colors.sequential.Greens
            self.colorscale = [[0, 'rgba(255,255,255,0)'], [0.1, color_list[1]], 
                                [0.5, color_list[3]], [1, color_list[5]]]
            self.scatter_color = color_list[5]  # Use a darker green for scatter points
        else:  # default to viridis
            self.colorscale = [[0, 'rgba(255,255,255,0)'], [0.1, 'rgba(68,1,84,0.3)'], 
                              [0.5, 'rgba(65,182,196,0.5)'], [1, 'rgba(253,231,37,0.8)']]
            self.scatter_color = 'rgba(68,1,84,0.8)'  # Use a dark purple for scatter points
        
        # Calculate the number of frames
        self.n_frames = max(1, (len(self.x_data) - self.window_size) // self.step_size + 1)
        
        # Create a figure
        self.fig = make_subplots(rows=1, cols=1)
        
        # Get data range for consistent axes - include both data and fixed points
        self._compute_axis_ranges()
        
        # Create frames for the animation
        self.frames = self._create_frames()

    def _compute_axis_ranges(self):
        """Compute axis ranges to include all data points and fixed points."""
        # Start with data ranges
        x_points = self.x_data
        y_points = self.y_data
        
        # Include fixed points in the range computation
        if self.fixed_points_matrix is not None and len(self.fixed_points_matrix) > 0:
            x_fixed = self.fixed_points_matrix[:, 0]
            y_fixed = self.fixed_points_matrix[:, 1]
            x_points = np.concatenate([x_points, x_fixed])
            y_points = np.concatenate([y_points, y_fixed])
        
        # Compute min/max with padding
        self.x_min, self.x_max = np.min(x_points), np.max(x_points)
        self.y_min, self.y_max = np.min(y_points), np.max(y_points)
        
        # Add padding (10% on each side)
        x_pad = 0.1 * (self.x_max - self.x_min)
        y_pad = 0.1 * (self.y_max - self.y_min)
        
        self.x_range = [self.x_min - x_pad, self.x_max + x_pad]
        self.y_range = [self.y_min - y_pad, self.y_max + y_pad]
        
        # Store the range info for potential normalization
        self.x_scale = self.x_max - self.x_min
        self.y_scale = self.y_max - self.y_min

    
    def _create_fixed_point_traces(self):
        """Create trace objects for fixed points to display in each frame."""
        fixed_point_traces = []
        
        # Process matrix-based fixed points (like A_matrix_2D)
        if self.fixed_points_matrix is not None:
            for idx, point in enumerate(self.fixed_points_matrix):
                # Get symbol and color (with defaults)
                symbol = self.point_symbols[idx] if idx < len(self.point_symbols) else 'circle'
                color = self.point_colors[idx] if idx < len(self.point_colors) else 'black'
                
                # Ensure point coordinates are within the same scale as the data points
                x_coord = point[0]
                y_coord = point[1]
                
                # Verify the point is within the computed range (or close to it)
                if x_coord < self.x_range[0] or x_coord > self.x_range[1] or \
                   y_coord < self.y_range[0] or y_coord > self.y_range[1]:
                    # Log a warning if point is significantly outside the range
                    print(f"Warning: Fixed point ({x_coord}, {y_coord}) may be outside the expected range.")
                
                trace = go.Scatter(
                    x=[x_coord],
                    y=[y_coord],
                    mode='markers',
                    marker=dict(
                        symbol=symbol,
                        size=18,
                        line_width=0.5,
                        color=color,
                        opacity=0.8,  # transparency
                    ),
                    showlegend=False,
                    hoverinfo='none'
                )
                
                fixed_point_traces.append(trace)
            
        return fixed_point_traces
    
    def _create_frames(self):
        """Create all the frames for the animation."""
        frames = []
        
        # Create fixed point traces once for all frames
        fixed_point_traces = self._create_fixed_point_traces()
        
        for frame_idx in range(self.n_frames):
            # Get window data
            start_idx = frame_idx * self.step_size
            end_idx = min(start_idx + self.window_size, len(self.x_data))
            
            x_window = self.x_data[start_idx:end_idx]
            y_window = self.y_data[start_idx:end_idx]

            # Calculate left and right edges as percentages (0-100%)
            left_percent = (start_idx / len(self.x_data)) * 100
            right_percent = (end_idx / len(self.x_data)) * 100
            # Calculate center position percentage
            center_percent = (left_percent + right_percent) / 2
            
            # Start with base traces - use the same trace indices as the initial figure
            frame_traces = [
                # Scatter plot - use uid=0 to match the initial trace
                go.Scatter(
                    x=x_window, 
                    y=y_window, 
                    mode='markers',
                    marker=dict(size=4, opacity=0.5, color=self.scatter_color),
                    showlegend=False,
                    hoverinfo='none',
                    uid='scatter-points'  # Consistent ID for animation transitions
                ),
            ]
            
            # Add fixed point traces to each frame with consistent IDs
            for i, trace in enumerate(fixed_point_traces):
                # Clone trace with a consistent ID
                fixed_trace = go.Scatter(
                    x=trace.x,
                    y=trace.y,
                    mode=trace.mode,
                    marker=trace.marker,
                    showlegend=trace.showlegend,
                    hoverinfo=trace.hoverinfo,
                    uid=f'fixed-point-{i}'  # Consistent ID for each fixed point
                )
                frame_traces.append(fixed_trace)
            
            # Create a frame with all traces, ensuring consistent axis ranges
            frame = go.Frame(
                data=frame_traces,
                name=str(frame_idx),
                layout=go.Layout(
                    xaxis=dict(
                        range=self.x_range,
                        showgrid=True,
                        gridcolor='rgba(200,200,200,0.2)',
                        gridwidth=1
                    ),
                    yaxis=dict(
                        range=self.y_range,
                        showgrid=True,
                        gridcolor='rgba(200,200,200,0.2)',
                        gridwidth=1
                    ),
                    annotations=[
                        dict(
                            text=f"Progress: {center_percent:.1f}%",
                            x=0.02,  # Position at the left side
                            y=0.98,  # Position at the top
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(
                                family="Arial",
                                size=14,
                                color=self.scatter_color
                            ),
                            bgcolor="rgba(255, 255, 255, 0.7)",
                            bordercolor="rgba(0, 0, 0, 0.2)",
                            borderwidth=1,
                            borderpad=4,
                            align="left"
                        )
                    ]
                )
            )
            frames.append(frame)
            
        return frames
    
    def visualize(self, play_interval=500):
        """
        Create the interactive sliding window contour visualization.
        
        Parameters:
        -----------
        play_interval : int
            Delay between frames in milliseconds.
        
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            The plotly figure object.
        """
        # Get first frame data for initial display
        start_idx = 0
        end_idx = min(self.window_size, len(self.x_data))
        x_window = self.x_data[start_idx:end_idx]
        y_window = self.y_data[start_idx:end_idx]
        
        # Calculate initial center_percent
        left_percent = (start_idx / len(self.x_data)) * 100
        right_percent = (end_idx / len(self.x_data)) * 100
        center_percent = (left_percent + right_percent) / 2
        
        # Initialize with the first frame data
        self.fig.add_trace(
            go.Scatter(
                x=x_window, 
                y=y_window, 
                mode='markers',
                marker=dict(size=4, opacity=0.5, color=self.scatter_color),
                showlegend=False,
                hoverinfo='none',
                uid='scatter-points'  # Consistent ID to match frames
            )
        )
        
        # Add fixed points to the initial display
        for i, trace in enumerate(self._create_fixed_point_traces()):
            # Add the trace with a consistent UID that will match frames
            fixed_trace = go.Scatter(
                x=trace.x,
                y=trace.y,
                mode=trace.mode,
                marker=trace.marker,
                showlegend=trace.showlegend,
                hoverinfo=trace.hoverinfo,
                uid=f'fixed-point-{i}'  # Consistent ID to match frames
            )
            self.fig.add_trace(fixed_trace)
        
        # Update layout
        self.fig.update_layout(
            width=self.plot_width,
            height=self.plot_height,
            margin=dict(l=0, r=0, t=0, b=0, pad=0),  # Remove margins for larger plot area
            xaxis=dict(
                range=self.x_range, 
                showticklabels=False,  # Hide tick labels
                showgrid=True,        # Show grid
                gridcolor='rgba(200,200,200,0.2)',  # Light gray grid
                gridwidth=1,
                zeroline=False,        # Hide zero line
                showline=False,        # Hide axis line
                autorange=False,       # Disable autorange to keep consistent scale
                constrain="domain"     # Keep aspect ratio consistent
            ),
            yaxis=dict(
                range=self.y_range, 
                showticklabels=False,  # Hide tick labels
                showgrid=True,        # Show grid
                gridcolor='rgba(200,200,200,0.2)',  # Light gray grid
                gridwidth=1,
                zeroline=False,        # Hide zero line
                showline=False,        # Hide axis line
                autorange=False,       # Disable autorange to keep consistent scale
                scaleanchor="x",       # Scale y-axis to match x-axis scale
                scaleratio=1           # Use 1:1 aspect ratio
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,          # Hide legend
            updatemenus=[
                {
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {
                            'label': '▶️ Play',
                            'method': 'animate',
                            'args': [
                                None, 
                                {
                                    'frame': {'duration': play_interval, 'redraw': False},
                                    'fromcurrent': True,
                                    'transition': {'duration': 300, 'easing': 'cubic-in-out'}
                                }
                            ]
                        },
                        {
                            'label': '⏸️ Pause',
                            'method': 'animate',
                            'args': [
                                [None], 
                                {
                                    'frame': {'duration': 0, 'redraw': False},
                                    'mode': 'immediate',
                                    'transition': {'duration': 0}
                                }
                            ]
                        },
                        {
                            'label': '🔄 Reset',
                            'method': 'relayout',
                            'args': [
                                {
                                    'xaxis.range': self.x_range,
                                    'yaxis.range': self.y_range
                                }
                            ]
                        }
                    ],
                    'x': 0.1,
                    'y': 0,
                    'xanchor': 'right',
                    'yanchor': 'top',
                    'pad': {'r': 10, 't': 10},
                    'bgcolor': 'rgba(255, 255, 255, 0.7)',
                    'bordercolor': 'rgba(0, 0, 0, 0.2)',
                    'borderwidth': 1
                }
            ],
            sliders=[
                {
                    'active': 0,
                    'yanchor': 'top',
                    'xanchor': 'left',
                    'currentvalue': {
                        'visible': False  # Hide current frame indicator
                    },
                    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
                    'pad': {'b': 0, 't': 0},
                    'len': 0.9,
                    'x': 0.1,
                    'y': 0,
                    'steps': [
                        {
                            'args': [
                                [frame.name],
                                {
                                    'frame': {'duration': play_interval, 'redraw': False},
                                    'mode': 'immediate',
                                    'transition': {'duration': 300, 'easing': 'cubic-in-out'}
                                }
                            ],
                            'label': str(i+1),
                            'method': 'animate'
                        }
                        for i, frame in enumerate(self.frames)
                    ]
                }
            ]
        )
        
        # Configure scene
        self.fig.update_layout(
            uirevision='true',  # Maintain user interaction states
            hovermode=False     # Disable hover interactions
        )
        
        # Add frames to the figure
        self.fig.frames = self.frames
        
        return self.fig

def animate_landscape(
    model_name: str = 'Meta-Llama-3.1-70B-Instruct-Turbo',
    dataset_name: str = 'aqua',
    method: str = None, # None for all methods
    plot_type: str = 'method',

    window_size: int = 300, # Number of points in each window
    step_size: int = 100, # How many points to advance in each frame

    save_root: str = "Landscape-Data",
    output_dir: str = "figures/animate_landscape",
    display: bool = False
):

    # Define point symbols and colors, we define this as the maximum number of answers in the dataset we adopted
    point_symbols = ['star', 'x', 'x', 'x', 'x']
    point_colors = ['green', 'red', 'red', 'red', 'red']

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
        (
            wrong_x, wrong_y, correct_x, correct_y
        ) = get_points_wrong_and_correct_chains(
            plot_datas=plot_datas,
            splited_T_2D=splited_T_2D,
            num_all_thoughts_w_start_list=num_all_thoughts_w_start_list,
        )

        '''
            Wrong chains
        '''
        # Create the visualizer with matrix-based fixed points
        visualizer = PlotlySlidingContourVisualizer(
            x_data=wrong_x, 
            y_data=wrong_y,
            window_size=window_size,  # Number of points in each window
            step_size=step_size,     # How many points to advance in each frame
            color_theme='red',  # Choose color theme: 'blue', 'red', 'green', or 'viridis'
            fixed_points_matrix=A_matrix_2D,  # Add fixed points from a matrix
            point_symbols=point_symbols,
            point_colors=point_colors,
            plot_width=900,   # Larger plot width
            plot_height=700   # Larger plot height
        )

        # Create and show the figure
        wrong_fig = visualizer.visualize(play_interval=500)
        pio.write_html(wrong_fig, file=os.path.join(output_dir, f"{model_name}-{dataset_name}-{methods[method_idx]}-wrong.html"))

        '''
            Correct chains
        '''
        # Create the visualizer with matrix-based fixed points
        visualizer = PlotlySlidingContourVisualizer(
            x_data=correct_x, 
            y_data=correct_y,
            window_size=window_size,  # Number of points in each window
            step_size=step_size,     # How many points to advance in each frame
            color_theme='blue',  # Choose color theme: 'blue', 'red', 'green', or 'viridis'
            fixed_points_matrix=A_matrix_2D,  # Add fixed points from a matrix
            point_symbols=point_symbols,
            point_colors=point_colors,
            plot_width=900,   # Larger plot width
            plot_height=700   # Larger plot height
        )

        # Create and show the figure
        correct_fig = visualizer.visualize(play_interval=500)
        pio.write_html(correct_fig, file=os.path.join(output_dir, f"{model_name}-{dataset_name}-{methods[method_idx]}-correct.html"))

        print(f"==> Saved animate landscape in {output_dir}/{model_name}-{dataset_name}-{methods[method_idx]}")
        method_idx += 1

        if display:
            wrong_fig.show()
            correct_fig.show()

if __name__ == "__main__":
    Fire(animate_landscape)