import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Dict, List, Any, Optional, Union

def visualize(features: Union[List, np.ndarray], metrics: Optional[Dict[str, Union[List, np.ndarray]]] = None, 
              file_pattern: str = "visualization_%d.png", title: str = "Landscape of Thoughts",
              figsize: tuple = (12, 8), dpi: int = 300):
    """
    Visualize the reasoning paths in a 2D space.
    
    Args:
        features: Features of the reasoning states. Can be a list of lists of lists or a 3D numpy array.
        metrics: Optional dictionary of metrics to visualize. Keys are metric names, values are lists or numpy arrays.
        file_pattern: Pattern for the output file names. Should include '%d' for the index.
        title: Title for the plot.
        figsize: Figure size.
        dpi: DPI for the output images.
    """
    # Convert features to numpy array if it's a list
    if isinstance(features, list):
        # Handle ragged arrays (different path lengths)
        max_path_length = max(len(path) for path in features)
        features_array = np.zeros((len(features), max_path_length, features[0][0].shape[0]))
        for i, path in enumerate(features):
            for j, state in enumerate(path):
                features_array[i, j] = state
    else:
        features_array = features
    
    # Reshape the features for t-SNE
    num_paths, max_path_length, feature_dim = features_array.shape
    features_reshaped = features_array.reshape(-1, feature_dim)
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features_reshaped)
    
    # Reshape back to paths
    features_2d = features_2d.reshape(num_paths, max_path_length, 2)
    
    # Create the plot
    plt.figure(figsize=figsize, dpi=dpi)
    
    # Plot each path
    for i, path in enumerate(features_2d):
        # Filter out zero padding
        if isinstance(features, list):
            path = path[:len(features[i])]
        
        # Plot the path
        plt.plot(path[:, 0], path[:, 1], '-o', alpha=0.7, linewidth=2, markersize=5)
        
        # Mark the start and end points
        plt.plot(path[0, 0], path[0, 1], 'o', markersize=8, color='green')
        plt.plot(path[-1, 0], path[-1, 1], 'o', markersize=8, color='red')
    
    # Add metrics as a legend if provided
    if metrics is not None:
        legend_text = []
        for metric_name, metric_values in metrics.items():
            if isinstance(metric_values, np.ndarray):
                metric_mean = np.mean(metric_values)
            else:
                metric_mean = sum(metric_values) / len(metric_values)
            legend_text.append(f"{metric_name}: {metric_mean:.4f}")
        
        plt.figtext(0.02, 0.02, "\n".join(legend_text), fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
    
    # Set plot properties
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.savefig(file_pattern % 0)
    plt.close()
    
    return features_2d
