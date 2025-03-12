# 🌌 Landscape of Thoughts (LoT)

A lightweight library for visualizing reasoning paths of Large Language Models (LLMs).

## 🚀 Installation

```bash
pip install -e .
```

## 📋 Requirements

LoT has minimal dependencies:

- numpy>=1.20.0
- scikit-learn>=1.0.0
- matplotlib>=3.4.0
- tqdm>=4.60.0

Optional dependencies for model access:

- openai (for vLLM local inference)
- together (for Together AI API)
- requests (for updating Together AI model information)

## 🔍 Overview

Landscape of Thoughts (LoT) is a library that allows you to visualize the reasoning paths of LLMs through dimensional projection. It provides a simple interface to:

1. Sample reasoning paths from LLMs using different algorithms
2. Extract features from these paths
3. Visualize the paths in a 2D space

## 🛠️ Usage

### Basic Example

```python
import lot

# Load a dataset
dataset = lot.datasets.AQuA("data/aqua.jsonl")

# Create a model (using Together AI API)
model = lot.models.Llama3(api_key="your_api_key")

# Or use a local model with vLLM
# model = lot.models.LocalModel("meta-llama/Meta-Llama-3.1-8B-Instruct", port=8000)

# Create an algorithm
algorithm = lot.algorithms.ToT()

# Sample reasoning paths
features, metrics = lot.sample(dataset, model, algorithm, num_sample=10)

# Visualize the paths
lot.visualize(features, metrics=metrics, file_pattern="visualization_%d.png")
```

### Together AI Model Management

#### Automatic Model List Updates

When using Together AI models, the library automatically checks if the model list is outdated (older than 7 days) and updates it if necessary. You can control this behavior with the `auto_update_models` parameter:

```python
# Disable automatic model list updates
model = lot.models.Llama3(api_key="your_api_key", auto_update_models=False)

# Force an update of the model list
from lot.models.unified_model import APIModel
APIModel.update_model_list_if_needed(api_key="your_api_key", force=True)
```

#### Manual Model List Updates

You can also manually update the Together AI model list using the provided script:

```bash
# Update the model information using the API key from the environment
python -m lot.models.update_together_models

# Or provide the API key directly
python -m lot.models.update_together_models --api-key your_api_key

# Specify a custom output path
python -m lot.models.update_together_models --output custom_path.json
```

You can also use the script programmatically:

```python
from lot.models.update_together_models import get_together_models, save_model_info

# Fetch the models
models = get_together_models(api_key="your_api_key")

# Save the model information
save_model_info(models, "models/data/togetherAI_model_infos.json")
```

### Using Your Own Model

You can use your own model with LoT in several ways:

#### Option 1: Using the CustomModel wrapper

```python
from lot.models import CustomModel

# Wrap an existing model
my_model = CustomModel(
    model=your_existing_model,  # Your existing model instance
)

# Or provide custom functions
def generate_text(prompt):
    # Your custom generation logic
    return "Generated response"

def calculate_likelihood(prompt, continuation):
    # Your custom likelihood calculation
    return 0.7

my_model = CustomModel(
    generate_fn=generate_text,
    likelihood_fn=calculate_likelihood
)
```

#### Option 2: Implementing the BaseModel interface

```python
from lot.models import BaseModel

class MyCustomModel(BaseModel):
    def __init__(self):
        # Initialize your model
        pass

    def __call__(self, prompt: str) -> str:
        # Generate a response for the prompt
        return "Your model's response"

    def get_likelihood(self, prompt: str, continuation: str) -> float:
        # Calculate the likelihood of the continuation given the prompt
        return 0.5

    def get_perplexity(self, prompt: str, continuation: str) -> float:
        # Calculate the perplexity of the continuation given the prompt
        return 1.0
```

### Using Your Own Dataset

You can use your own dataset with LoT by implementing the `BaseDataset` interface:

```python
from lot.datasets import BaseDataset

class CustomDataset(BaseDataset):
    def __init__(self):
        # Initialize your dataset
        self.data = [{"query": "Question 1", "answer": "Answer 1"}, ...]

    def __getitem__(self, idx: int):
        # Return a single example
        return self.data[idx]

    def __len__(self):
        # Return the number of examples
        return len(self.data)

    def __iter__(self):
        # Return an iterator over the examples
        return iter(self.data)
```

### Using JSON Datasets

LoT provides a convenient `JsonDataset` class for loading JSON/JSONL datasets:

```python
dataset = lot.datasets.JsonDataset(
    data_path="data/custom.jsonl",
    query_field="question",
    answer_field="answer",
    is_jsonl=True
)
```

## 📊 Visualization

LoT uses t-SNE to project reasoning paths into a 2D space. The visualization shows:

- Each reasoning path as a connected line
- Start points in green
- End points in red
- Optional metrics as a legend

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
