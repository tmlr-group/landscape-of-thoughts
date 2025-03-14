# Trace of Thoughts (LOT) Examples

This repository contains examples of how to use the `lot` module to sample reasoning traces from different datasets using different models and algorithms.

## Installation

First, install the `lot` module:

```bash
pip install -e .
```

## Usage

### Sampling Reasoning Traces

The `sample_reasoning_trace_with_lot.py` script demonstrates how to use the `lot` module to sample reasoning traces from different datasets using different models and algorithms.

```bash
python sample_reasoning_trace_with_lot.py \
    --model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" \
    --port=8000 \
    --dataset_name="mmlu" \
    --method="cot" \
    --samples=10 \
    --start_index=0 \
    --end_index=50
```

#### Parameters

- `model_name`: Name of the model to use (default: 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo')
- `port`: Port for the model API (default: 8000)
- `dataset_name`: Name of the dataset to use (default: 'mmlu')
- `method`: Method to use for reasoning (cot, tot, mcts) (default: 'cot')
- `samples`: Number of samples per query (default: 10)
- `start_index`: Start index in the dataset (default: 0)
- `end_index`: End index in the dataset (default: 50)

### Custom Model Example

The `custom_model_example.py` script demonstrates how to wrap a custom model and dataset for use with the `lot` module.

```bash
python custom_model_example.py \
    --dataset_path="data/custom_dataset.json" \
    --method="cot" \
    --samples=5
```

#### Parameters

- `dataset_path`: Path to the custom dataset (default: 'data/custom_dataset.json')
- `method`: Method to use for reasoning (cot, tot) (default: 'cot')
- `samples`: Number of samples per query (default: 5)

## Creating a Custom Model

To create a custom model, you need to implement a class that inherits from `lot.models.base.BaseModel` and implements the `__call__` and `get_likelihood` methods:

```python
class CustomModel(lot.models.base.BaseModel):
    def __init__(self, model_name="your-custom-model", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        # Initialize your custom model here

    def __call__(self, prompt):
        # Implement your custom model's generation logic here
        return "Generated response"

    def get_likelihood(self, prompt, continuation):
        # Implement your custom model's likelihood calculation logic here
        return 0.8  # Likelihood between 0 and 1
```

## Creating a Custom Dataset

To create a custom dataset, you need to implement a class that inherits from `lot.datasets.base.BaseDataset` and implements the `__getitem__` and `__len__` methods:

```python
class CustomDataset(lot.datasets.base.BaseDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(**kwargs)
        self.data_path = data_path
        # Load your custom dataset here

    def __getitem__(self, idx):
        # Return the item at the given index
        return self.data[idx]

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)
```

## Visualization

The `lot.visualize` function can be used to visualize the reasoning traces:

```python
lot.visualize(features, metrics=metrics, file_pattern="visualization_%d.png")
```

This will save the visualization to files named according to the pattern (e.g., "visualization_0.png", "visualization_1.png", etc.).

## Directory Structure

- `data/`: Contains the datasets
- `prompts/`: Contains the prompt templates for different algorithms and datasets
- `exp-data-scale/`: Contains the results of sampling reasoning traces using scale-based methods (e.g., CoT)
- `exp-data-searching/`: Contains the results of sampling reasoning traces using search-based methods (e.g., ToT, MCTS)
- `exp-data-custom/`: Contains the results of sampling reasoning traces using custom models and datasets
