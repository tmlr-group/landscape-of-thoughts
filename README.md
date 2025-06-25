<div align="center">

<h1>Landscape of Thoughts</h1>
<h3>Visualizing the Reasoning Process of Large Language Models</h3>

[![Project Website](https://img.shields.io/badge/Project%20Website-deepinception)](https://landscape-of-thoughts.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-2503.22165-b31b1b)](https://arxiv.org/abs/2503.22165)
[![Hugging Face Datasets](https://img.shields.io/badge/Datasets-blue)](https://huggingface.co/datasets/GazeEzio/Landscape-Data)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IgLREaEw-FeJbKn9NfYGIyaex2QhgCT2?usp=sharing)

<div align="center">
    <figure>
        <img src="assets/demo.png" alt="Demo of Landscape of Thoughts Visualization">
        <figcaption style="text-align: left;">Illustration of Landscape of thoughts for visualizing the reasoning steps of LLMs. Note that the <span style="color: red;">red</span> landscape represents wrong reasoning cases, while the <span style="color: lightblue;">blue</span> indicates the correct ones. The darker regions in landscapes indicate more thoughts, with <img src="./assets/red_cross.png" style="height:1.0em; vertical-align:middle;"> indicating incorrect answers and <img src="./assets/green_star.png" style="height:1.0em; vertical-align:middle;"> marking correct answers. </figcaption>
    </figure>
</div>

</div>

## Motivation

Large language models (LLMs) increasingly rely on step-by-step reasoning for various applications, yet their reasoning processes remain poorly understood, hindering research, development, and safety efforts. Current approaches to analyze LLM reasoning lack comprehensive visualization tools that can reveal the internal structure and patterns of reasoning paths.

To address this challenge, we introduce **Landscape of Thoughts**, the first visualization framework designed to explore the reasoning paths of chain-of-thought and its derivatives across any multiple-choice dataset. Our approach represents reasoning states as feature vectors, capturing their distances to all answer choices, and visualizes them in 2D using t-SNE dimensionality reduction.

### Key Capabilities

Through qualitative and quantitative analysis, Landscape of Thoughts enables researchers to:

- **Distinguish model performance**: Effectively differentiate between strong versus weak models
- **Analyze reasoning quality**: Compare correct versus incorrect reasoning paths
- **Explore task diversity**: Understand reasoning patterns across different types of problems
- **Identify failure modes**: Reveal undesirable reasoning patterns such as inconsistency and high uncertainty

## Installation

We provide two installation methods:

### Package Installation (Recommended)

Install the Landscape of Thoughts framework directly via pip:

```bash
pip install landscape-of-thoughts==0.1.0
```

### Development Installation

For development or customization, clone the repository and set up the environment:

```bash
# Clone the repository
git clone https://github.com/tmlr-group/landscape-of-thoughts.git
cd landscape-of-thoughts

# Create and activate conda environment
conda create -n landscape python=3.10
conda activate landscape

# Install dependencies
pip install -r requirements.txt
pip install fire --use-pep517
```

### Model Setup

Before analyzing your data, you need to set up a language model. For detailed instructions, see our [model setup guidance](doc/setup_model.md).

## Usage

Two ways to use the framework to plot the landscape, depending on the installation method:

- If you installed the package, you can use the `lot` command to plot the landscape.
- If you installed the framework from source, you can use the `main.py` script to plot the landscape.

### Quick Start with Command Line Interface

After installing the package and setting up your model, you can start analyzing reasoning patterns immediately:

For example, the following command executes the complete analysis pipeline. It employs the `meta-llama/Llama-3.2-1B-Instruct` model to generate 10 reasoning traces for each of the first 5 examples in the `AQUA` dataset, using the Chain-of-Thought (`cot`) method. The model is hosted locally (`--local`) via vLLM with the API key `token-abc123`. Finally, it generates and saves the landscape visualization in the `figures/landscape`. More configuration options are available in the [configuration section](#configuration).

```bash
lot --task all \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --dataset_name aqua \
    --method cot \
    --num_samples 10 \
    --start_index 0 \
    --end_index 5 \
    --output_dir figures/landscape \
    --local \
    --local_api_key token-abc123
```

### Unified Script Interface

Use the main script for complete pipeline execution:

```bash
python main.py \
  --task all \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --dataset_name aqua \
  --method cot \
  --num_samples 10 \
  --start_index 0 \
  --end_index 5 \
  --output_dir figures/landscape \
  --local \
  --local_api_key token-abc123
```

### Task Control

For advanced usage and integration into research workflows, you can utilize the Python API. The `task` parameter allows you to control which components of the pipeline to execute:

- `sample`: This option generates reasoning traces from the language model.
- `calculate`: This option computes distance matrices between reasoning states.
- `plot`: This option creates visualizations of the reasoning landscape.
- `all`: This option executes the complete all three tasks.

The following example demonstrates how to use the API to perform each step of the analysis pipeline individually.

1. `sample` generates 10 reasoning traces for the first 5 examples of the `AQUA` dataset using the `meta-llama/Meta-Llama-3-8B-Instruct` model and the CoT method.
2. `calculate` computes the distance matrices for these traces.
3. `plot` generates the landscape visualization from the processed data.

```python
from lot import sample, calculate, plot

# Generate reasoning traces
features, metrics = sample(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    dataset_name="aqua",
    method="cot",
    num_samples=10,
    start_index=0,
    end_index=5
)

# Calculate distance matrices
distance_matrices = calculate(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    dataset_name="aqua",
    method="cot",
    start_index=0,
    end_index=5
)

# Generate visualizations
plot(
    model_name="Meta-Llama-3-8B-Instruct",
    dataset_name="aqua",
    method="cot",
)
```

### Animation Visualization

The example below shows how to generate an animation for the `Meta-Llama-3.1-70B-Instruct-Turbo` model on the `AQUA` dataset using the CoT method. The animation will be saved to the `figures/animation` directory. Note that the `Landscape-Data` is pulled from the [Landscape-Data](https://huggingface.co/datasets/GazeEzio/Landscape-Data) dataset.

```python
from lot.animation import animation_plot
from datasets import load_dataset

# This will download and cache the dataset in the "Landscape-Data" directory
dataset = load_dataset("GazeEzio/Landscape-Data", cache_dir="Landscape-Data")

animation_plot(
    model_name = 'Meta-Llama-3.1-70B-Instruct-Turbo',
    dataset_name = 'aqua',
    method = 'cot',
    save_root = "Landscape-Data",
    save_video = True,
    output_dir = "figures/animation",
)
```

For detailed examples, see [animation.ipynb](./animation.ipynb).

## Configuration

### Key Parameters

- `model_name`: Identifier for the language model (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`)
- `dataset_name`: Target dataset for analysis (e.g., `aqua`, `mmlu`)
- `method`: Reasoning approach (`cot`, `tot`, `mcts`, `l2m`)
- `num_samples`: Number of reasoning traces to collect per example
- `start_index`/`end_index`: Range of dataset examples to process

### Supported Models

The framework supports any open-source language model accessible via API, provided that token-level log probabilities are available. Models can be hosted using:

- **vLLM**: For local model serving
- **API providers**: Compatible with OpenAI-style APIs

#### Example: Using Qwen2.5-3B-Instruct

1. Host the model locally using vLLM:

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --api-key "token-api-123" \
  --download-dir YOUR_MODEL_PATH \
  --port 8000
```

2. Run analysis with the hosted model:

```bash
python main.py \
  --task all \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --dataset_name aqua \
  --method cot \
  --num_samples 10 \
  --start_index 0 \
  --end_index 5 \
  --plot_type method \
  --output_dir figures/landscape \
  --local \
  --local_api_key token-abc123
```

### Supported Datasets

The framework accepts any multiple-choice question dataset in JSONL format with the following structure:

```json
{
  "question": "What is the capital of France?",
  "options": ["A) London", "B) Berlin", "C) Paris", "D) Madrid"],
  "answer": "C"
}
```

#### Built-in Datasets

- `aqua`: Algebraic reasoning problems
- `commonsenseqa`: Common sense reasoning questions
- `mmlu`: Massive multitask language understanding
- `strategyqa`: Strategic reasoning questions

#### Custom Datasets

Create your own datasets following our format specifications. For detailed instructions on creating, validating, and using custom datasets, see our [Custom Datasets Guidance](./doc/custom_datasets.md).

## Advanced Features

### Reasoning Methods

- **Chain-of-Thought (CoT)**: Step-by-step sequential reasoning
- **Tree-of-Thoughts (ToT)**: Exploration of multiple reasoning branches
- **Monte Carlo Tree Search (MCTS)**: Strategic search through reasoning paths
- **Least-to-Most (L2M)**: Decomposes complex problems into a sequence of simpler subproblems

### Visualization Types

- **Method comparison**: Compare different reasoning approaches
- **Correctness analysis**: Distinguish correct vs. incorrect reasoning
- **Task analysis**: Explore reasoning patterns across problem types
- **Temporal dynamics**: Animate reasoning progression over reasoning steps

## Examples and Tutorials

- [Dataset and Model Usage](examples/dataset_and_model_usage.py)
- [Custom Model Integration](examples/custom_model_example.py)
- [Dataset Creation](examples/dataset_example.py)
- [Prompt Customization](examples/prompt_example.py)
- [Animation Tutorial](animation.ipynb)
- [Quick Start Notebook](quick_start.ipynb)

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{zhou2025landscape,
  title={Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models},
  author={Zhanke Zhou and Zhaocheng Zhu and Xuan Li and Mikhail Galkin and Xiao Feng and Sanmi Koyejo and Jian Tang and Bo Han},
  journal={arXiv preprint arXiv:2503.22165},
  year={2025},
  url={https://arxiv.org/abs/2503.22165}
}
```

## Contact

For questions, technical support, or collaboration inquiries:

- Email: cszkzhou@comp.hkbu.edu.hk
- Issues: [GitHub Issues](https://github.com/tmlr-group/landscape-of-thoughts/issues)
