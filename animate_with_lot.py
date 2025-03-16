import os
import plotly.io as pio
from fire import Fire
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure Plotly to not automatically open browser windows
pio.renderers.default = None

from lot.visualization import create_animation_from_file


def main(
    model_name: str = 'Meta-Llama-3-8B-Instruct-Lite',
    dataset_name: str = 'aqua',
    method: str = 'cot',
    sample_idx: int = 0,
    tool: str = 'tsne',
    use_contour: bool = True,
    num_frames: int = 20,
    unified_contours: bool = True,
    save_root: str = "exp-data",
    output_dir: str = "figures/animation",
    save_html: bool = True,
    save_png: bool = False,
    display: bool = False
):
    """
    Main function to create animated visualizations of reasoning traces.
    
    Args:
        model_name (str): Name of the model to use.
        dataset_name (str): Name of the dataset to use.
        method (str): The reasoning method used (e.g., 'cot', 'standard').
        sample_idx (int): Index of the sample to animate.
        tool (str): Tool to use for dimensionality reduction ('tsne' or 'umap').
        use_contour (bool): Whether to use Histogram2dContour for visualization.
        num_frames (int): Number of frames for the animation.
        unified_contours (bool): Whether to unify correct and incorrect contours in a single visualization.
        save_root (str): Root directory where data is stored.
        output_dir (str): Directory to save output figures.
        save_html (bool): Whether to save the animation as an HTML file.
        save_png (bool): Whether to save a static image as a PNG file.
        display (bool): Whether to display the figure interactively (may require pressing 'q' to exit).
    """
    print(f"==> model_name: {model_name}")
    print(f"==> dataset_name: {dataset_name}")
    print(f"==> method: {method}")
    print(f"==> sample_idx: {sample_idx}")
    print(f"==> tool: {tool}")
    print(f"==> use_contour: {use_contour}")
    print(f"==> num_frames: {num_frames}")
    print(f"==> unified_contours: {unified_contours}")
    print(f"==> save_root: {save_root}")
    print(f"==> output_dir: {output_dir}")
    print(f"==> display: {display}")
    
    # Construct file paths
    thoughts_file = os.path.join(save_root, dataset_name, "thoughts", f"{model_name}--{method}--{dataset_name}--{sample_idx}.json")
    distance_matrix_file = os.path.join(save_root, dataset_name, "distance_matrix", f"{model_name}--{method}--{dataset_name}--{sample_idx}.pkl")
    
    # Check if files exist
    if not os.path.exists(thoughts_file):
        raise FileNotFoundError(f"Thoughts file not found: {thoughts_file}")
    if not os.path.exists(distance_matrix_file):
        raise FileNotFoundError(f"Distance matrix file not found: {distance_matrix_file}")
    
    print(f"==> Creating animation from files:")
    print(f"    - Thoughts file: {thoughts_file}")
    print(f"    - Distance matrix file: {distance_matrix_file}")
    
    # Create animation
    fig = create_animation_from_file(
        thoughts_file=thoughts_file,
        distance_matrix_file=distance_matrix_file,
        tool=tool,
        use_contour=use_contour,
        num_frames=num_frames,
        unified_contours=unified_contours
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the animation
    if save_html:
        html_path = os.path.join(output_dir, f"{model_name}-{dataset_name}-{method}-{sample_idx}.html")
        print(f"==> Saving animation to: {html_path}")
        fig.write_html(html_path)
        print(f"==> Animation saved to: {html_path}")
    
    if save_png:
        png_path = os.path.join(output_dir, f"{model_name}-{dataset_name}-{method}-{sample_idx}.png")
        print(f"==> Saving static image to: {png_path}")
        pio.write_image(fig, png_path, scale=3, width=900, height=600)
        print(f"==> Static image saved to: {png_path}")
    
    # Only display the figure if explicitly requested
    if display:
        print("==> Displaying figure (press 'q' to exit)")
        fig.show()
    
    print("==> Animation creation complete!")
    return fig


if __name__ == "__main__":
    Fire(main) 