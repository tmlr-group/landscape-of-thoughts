# Datasets Module

This module provides a unified interface for loading and working with various datasets in the Landscape of Thoughts (LoT) library.

## Unified Dataset Interface

The datasets module now provides a unified interface for loading datasets through the `get_dataset` function. This allows you to load any supported dataset using a consistent API, rather than having to import and instantiate specific dataset classes.

### Basic Usage

```python
from lot.datasets import get_dataset

# Load a CommonsenseQA dataset
commonsenseqa_dataset = get_dataset("commonsenseqa", "path/to/commonsenseqa.jsonl")

# Load a StrategyQA dataset
strategyqa_dataset = get_dataset("strategyqa", "path/to/strategyqa.json")

# Load a generic JSON dataset with custom field names
custom_dataset = get_dataset("json", "path/to/custom_dataset.json",
                            query_field="question",
                            answer_field="correct_answer",
                            is_jsonl=False)
```

### Available Datasets

You can list all available datasets using the `list_available_datasets` function:

```python
from lot.datasets import list_available_datasets

available_datasets = list_available_datasets()
print("Available datasets:")
for name, dataset_class in available_datasets.items():
    print(f"  - {name}: {dataset_class.__name__}")
```

Currently supported datasets:

- `aqua`: AQuA dataset
- `commonsenseqa`: CommonsenseQA dataset
- `mmlu`: MMLU dataset
- `strategyqa`: StrategyQA dataset
- `json`: Generic JSON/JSONL dataset

### Registering Custom Datasets

You can register your own custom dataset classes to use with the unified interface:

```python
from lot.datasets import BaseDataset, register_dataset

# Define a custom dataset class
class MyCustomDataset(BaseDataset):
    def __init__(self, data_path, **kwargs):
        # Initialize your dataset here
        self.data = [...]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

# Register the custom dataset
register_dataset("my_custom", MyCustomDataset)

# Now you can load it using the unified interface
dataset = get_dataset("my_custom", "path/to/data")
```

## Dataset Format

All datasets in the LoT library follow a consistent format. Each dataset item is a dictionary with at least the following keys:

- `query`: The question or prompt text
- `answer`: The correct answer

Different datasets may include additional fields specific to their format.

## Examples

See the `examples.py` file for more detailed examples of how to use the unified dataset interface.
