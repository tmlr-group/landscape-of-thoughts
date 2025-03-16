import os
import plotly.io as pio

from .visualization_utils import draw_landscape, process_landscape_data

def plot(
    model_name: str = 'Meta-Llama-3-8B-Instruct-Lite',
    dataset_name: str = 'aqua',
    method: str = 'cot',
    plot_type: str = 'method',
    save_root: str = "exp-data",
    output_dir: str = "figures/landscape"
) -> bool:
    """
    Main function to plot landscape visualizations of reasoning traces.
    
    Args:
        model_name (str): Name of the model to use.
        dataset_name (str): Name of the dataset to use.
        method (str): The reasoning method used (e.g., 'cot', 'standard').
        plot_type (str): Type of plot ('method' or 'model').
        save_root (str): Root directory where data is stored.
        output_dir (str): Directory to save output figures.
        
    Returns:
        bool: True if plotting was successful.
    """
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
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save plots
    method_idx = 0
    for plot_datas, splited_T_2D, num_all_thoughts_w_start_list in zip(list_plot_data, list_all_T_2D, list_num_all_thoughts_w_start_list):
        # Create the figure
        fig = draw_landscape(
            dataset_name=dataset_name,
            plot_datas=plot_datas,
            splited_T_2D=splited_T_2D,
            A_matrix_2D=A_matrix_2D,
            num_all_thoughts_w_start_list=num_all_thoughts_w_start_list
        )
        
        # Define save path
        save_path = os.path.join(output_dir, f"{model_name}-{dataset_name}-{methods[method_idx]}.png")
        
        # Increment method index if not specific method
        if not method:
            method_idx += 1
        
        # Save the figure
        print(f"==> Saving figure to: {save_path}")
        pio.write_image(fig, save_path, scale=6, width=1500, height=350)
    
    print("==> Plotting complete!")
    return True 