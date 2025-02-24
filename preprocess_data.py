from fire import Fire
from joblib import dump, load

from utils.train_utils import prepare_data_for_training


def generate_dataset_configs(
    start_idx=0,
    end_idx=5,
    specific_combinations=None
):
    """
    Generate dataset configurations with predefined options
    """
    # Predefined options
    DATASETS = ['aqua', 'mmlu', 'commonsenseqa', 'strategyqa'] # 'aqua', 'mmlu', 'commonsenseqa', 'strategyqa'
    MODELS = [
        # 'Llama3.2-1B-Instruct',
        # 'Llama3.2-3B-Instruct',
        'Llama3.1-8B-Instruct',
        'Meta-Llama-3.1-70B-Instruct-Turbo'
    ]
    METHODS = ['cot', 'l2m', 'zero_shot_cot'] # 'mcts', 'tot', 
    
    configs = []
    
    if specific_combinations:
        # Use specific combinations if provided
        for combo in specific_combinations:
            config = {
                'dataset': combo['dataset'],
                'model': combo['model'],
                'method': combo['method'],
                'start_idx': start_idx,
                'end_idx': end_idx
            }
            configs.append(config)
    else:
        # Generate all possible combinations
        for dataset in DATASETS:
            for model in MODELS:
                for method in METHODS:
                    config = {
                        'dataset': dataset,
                        'model': model,
                        'method': method,
                        'start_idx': start_idx,
                        'end_idx': end_idx
                    }
                    configs.append(config)
    
    return configs

def main(start_idx=0, end_idx=20):
    
    # 1. Generate all combinations (48 combinations total: 4 datasets × 4 models × 3 methods)
    dataset_configs = generate_dataset_configs(start_idx=start_idx, end_idx=end_idx)

    X, y, x_scaler, y_scaler, acc_infos = prepare_data_for_training(
        dataset_configs,
        verbose=False,
        mode='reg'
    )
    dataset = 'llama8B'

    dump([X, y, x_scaler, y_scaler, acc_infos], f"./training_data/processed_train_data/abl_{dataset}_cfgs_start-{start_idx}_end-{end_idx}.pkl")

if __name__ == "__main__":
    Fire(main)
