import os
import sys
from typing import Optional
from fire import Fire

from lot import sample, calculate, plot

def main(
    task: str = 'all',
    model_name: str = 'meta-llama/Meta-Llama-3-8B-Instruct-Lite',
    port: int = 8000,
    dataset_name: str = 'dummy',
    data_path: str = './lot/data/dummy.jsonl',
    method: str = 'cot',
    num_samples: int = 5,
    start_index: int = 0,
    end_index: int = 4,
    prompt_file: Optional[str] = None,
    max_tokens: int = 2048,
    plot_type: str = 'method',
    save_root: str = "demo_data",
    output_dir: str = "figures/landscape",
    local: bool = False,
    local_api_key: str = "token-abc123",
    options_field: str = "options",
    question_field: str = "question",
    **kwargs
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
        output_dir (str): Directory for output figures.
        local (bool): Whether to use local server.
        local_api_key (str): API key for the local server.
        options_field (str): Field name for the options in the dataset.
        question_field (str): Field name for the question in the dataset.
        **kwargs: Additional keyword arguments passed from command line.
    """
    # Check if help flag is provided
    if '--help' in sys.argv or '-h' in sys.argv:
        print("Landscape of Thoughts (lot) - A tool for visualizing LLM reasoning processes\n")
        print("Usage: lot [OPTIONS] [ARGS]...\n")
        print("Options:")
        print("  --task TEXT               Task to perform: 'sample', 'calculate', 'plot', or 'all'")
        print("  --model_name TEXT         Name of the model to use")
        print("  --port INTEGER            Port for the API server")
        print("  --dataset_name TEXT       Name of the dataset to use")
        print("  --data_path TEXT          Path to the dataset file")
        print("  --method TEXT             Method for reasoning (cot, standard, tot, mcts)")
        print("  --num_samples INTEGER     Number of samples per example")
        print("  --start_index INTEGER     Index of the first example to process")
        print("  --end_index INTEGER       Index of the last example to process")
        print("  --prompt_file TEXT        Path to a prompt file")
        print("  --max_tokens INTEGER      Maximum tokens for model responses")
        print("  --plot_type TEXT          Type of plot ('method', 'model')")
        print("  --save_root TEXT          Root directory to save results")
        print("  --output_dir TEXT         Directory for output figures")
        print("  --local                   Whether to use local server")
        print("  --local_api_key TEXT      API key for local server")
        print("  --options_field TEXT      Field name for the options in the dataset")
        print("  --question_field TEXT     Field name for the question in the dataset")
        print("  --help                    Show this help message and exit")
        return True
    
    # Print received arguments for debugging
    print(f"RECEIVED ARGUMENTS:")
    print(f"==> task: {task}")
    print(f"==> model_name: {model_name}")
    print(f"==> dataset_name: {dataset_name}")
    print(f"==> data_path: {data_path}")
    print(f"==> method: {method}")
    print(f"==> num_samples: {num_samples}")
    print(f"==> start_index: {start_index}")
    print(f"==> end_index: {end_index}")
    print(f"==> save_root: {save_root}")
    if kwargs:
        print("==> Additional arguments:")
        for key, value in kwargs.items():
            print(f"====> {key}: {value}")
    
    # Validate task
    valid_tasks = ['sample', 'calculate', 'plot', 'all']
    if task not in valid_tasks:
        raise ValueError(f"Invalid task: {task}. Must be one of {valid_tasks}")
    
    # Prepare common keyword arguments for all tasks
    common_kwargs = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'data_path': data_path,
        'method': method,
        'start_index': start_index,
        'end_index': end_index,
        'save_root': save_root,
        'local': local,
        'local_api_key': local_api_key
    }
    
    # Add any additional kwargs passed from command line
    common_kwargs.update(kwargs)
    
    # Run the specified task(s)
    if task == 'sample' or task == 'all':
        print("="*50)
        print("RUNNING SAMPLING TASK")
        print("="*50)
        
        # Add sample-specific arguments
        sample_kwargs = common_kwargs.copy()
        sample_kwargs.update({
            'port': port,
            'num_samples': num_samples,
            'prompt_file': prompt_file,
            'max_tokens': max_tokens,
            'options_field': options_field,
            'question_field': question_field
        })
        
        sample(**sample_kwargs)
    
    if task == 'calculate' or task == 'all':
        print("="*50)
        print("RUNNING CALCULATION TASK")
        print("="*50)
        
        # Use common arguments for calculate task
        calculate_kwargs = common_kwargs.copy()
        
        calculate(**calculate_kwargs)
    
    if task == 'plot' or task == 'all':
        print("="*50)
        print("RUNNING PLOTTING TASK")
        print("="*50)
        
        # Extract model name without path for plotting
        model_name_short = model_name.split("/")[-1] if "/" in model_name else model_name
        
        # Add plot-specific arguments
        plot_kwargs = {
            'model_name': model_name_short,
            'dataset_name': dataset_name,
            'method': method,
            'plot_type': plot_type,
            'save_root': save_root,
            'output_dir': output_dir
        }
        
        plot(**plot_kwargs)
    
    print("="*50)
    print("ALL TASKS COMPLETED")
    print("="*50)
    return True

if __name__ == "__main__":
    Fire(main) 