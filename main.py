import os
from typing import Optional
from fire import Fire
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lot import sample, calculate, plot

def main(
    task: str = 'all',
    model_name: str = 'meta-llama/Meta-Llama-3-8B-Instruct-Lite',
    port: int = 8000,
    dataset_name: str = 'aqua',
    data_path: str = './lot/data/aqua.jsonl',
    method: str = 'cot',
    num_samples: int = 10,
    start_index: int = 0,
    end_index: int = 2,
    prompt_file: Optional[str] = None,
    max_tokens: int = 2048,
    plot_type: str = 'method',
    save_root: str = "exp-data",
):
    """
    Main function to run the entire pipeline: sampling, calculating, and plotting.
    
    Args:
        task (str): Task to perform ('sample', 'calculate', 'plot', or 'all').
        model_name (str): Name of the model to use.
        port (int): Port for the API server.
        dataset_name (str): Name of the dataset to use.
        data_path (str): Path to the dataset file.
        method (str): Method to use for reasoning (cot, standard, tot, mcts).
        num_samples (int): Number of samples to generate per example.
        start_index (int): Index of the first example to process.
        end_index (int): Index of the last example to process.
        prompt_file (Optional[str]): Path to a prompt file for the algorithm.
        max_tokens (int): Maximum number of tokens for model responses.
        plot_type (str): Type of plot ('method', 'model',).
        save_root (str): Root directory to save results.
    """
    # Validate task
    valid_tasks = ['sample', 'calculate', 'plot', 'all']
    if task not in valid_tasks:
        raise ValueError(f"Invalid task: {task}. Must be one of {valid_tasks}")
    
    # Run the specified task(s)
    if task == 'sample' or task == 'all':
        print("="*50)
        print("RUNNING SAMPLING TASK")
        print("="*50)
        sample(
            model_name=model_name,
            port=port,
            dataset_name=dataset_name,
            data_path=data_path,
            method=method,
            num_samples=num_samples,
            start_index=start_index,
            end_index=end_index,
            prompt_file=prompt_file,
            max_tokens=max_tokens,
            save_root=save_root
        )
    
    # The sampling task generates reasoning traces from the LLM using the specified method.
    # It collects multiple reasoning paths for each example in the dataset, which will be
    # used for distance calculation and visualization in subsequent steps.
    if task == 'calculate' or task == 'all':
        print("="*50)
        print("RUNNING CALCULATION TASK")
        print("="*50)
        calculate(
            model_name=model_name,
            dataset_name=dataset_name,
            data_path=data_path,
            method=method,
            start_index=start_index,
            end_index=end_index,
            save_root=save_root
        )
    
    # The plotting task generates static visualizations of the reasoning paths.
    # It creates plots to analyze the performance of the reasoning method across
    # different examples or models.
    if task == 'plot' or task == 'all':
        print("="*50)
        print("RUNNING PLOTTING TASK")
        print("="*50)
        # Extract model name without path for plotting
        model_name_short = model_name.split("/")[-1] if "/" in model_name else model_name
        plot(
            model_name=model_name_short,
            dataset_name=dataset_name,
            method=method,
            plot_type=plot_type,
            save_root=save_root,
        )
    
    print("="*50)
    print("ALL TASKS COMPLETED")
    print("="*50)
    return True

if __name__ == "__main__":
    Fire(main)