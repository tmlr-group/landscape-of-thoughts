<div align="center">

<h1>Landscape of Thoughts</h1>
<h3>Visualizing the Reasoning Process of Large Language Models</h3>

[![Website](https://img.shields.io/badge/Project%20Website-landscapeofthoughts)](https://landscape-of-thoughts.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-2503.22165-b31b1b)](https://arxiv.org/abs/2503.22165)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-blue)](https://huggingface.co/datasets/GazeEzio/Landscape-Data)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IgLREaEw-FeJbKn9NfYGIyaex2QhgCT2?usp=sharing)

|     ![demo](assets/demo.png)     |
| :------------------------------: |
| Diagram of Landscape of Thoughts |

</div>

---

> [!NOTE]
> Before starting to analyze your own data, you may need to set up the model as described in [setup model](doc/setup_model.md).

## 📋 Overview

Landscape of Thoughts (LoT) is a framework for visualizing and analyzing the reasoning paths of Large Language Models (LLMs). This library provides tools to:

1. Sample reasoning traces from LLMs using various methods (CoT, ToT, MCTS)
2. Calculate distances between reasoning steps
3. Visualize the reasoning landscape through dimensional projection

## 🐍 Setting up Environment

```bash
# Create environment
conda create -n landscape python=3.10
pip3 install -r requirements.txt
```

## 🚄 Simplified API

You can use our unified script (`main.py`) that combines all three steps into a single command:

```bash
python main.py \
  --task all \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
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

The `task` parameter can be set to:

- `sample`: Only run the sampling step
- `calculate`: Only run the calculation step
- `plot`: Only run the visualization step
- `all`: Run the complete pipeline

This unified approach simplifies the workflow by handling all steps with consistent parameters and proper sequencing.

## 🧩 Library Usage

For more advanced usage, you can directly import functions from the `lot` package or use the main function:

```python
from lot import sample, calculate, plot

# Sample reasoning traces
features, metrics = sample(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    dataset_name="aqua",
    method="cot",
    num_samples=10,
    start_index=0,
    end_index=5
)

# Calculate distance matrices
distance_matrices = calculate(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    dataset_name="aqua",
    method="cot",
    start_index=0,
    end_index=5
)

# Generate visualizations
plot(
    model_name="Meta-Llama-3-8B-Instruct-Lite",
    dataset_name="aqua",
    method="cot",
)
```

## 🔧 Key Parameters

- `model_name`: Name of the LLM to use (e.g., meta-llama/Meta-Llama-3-8B-Instruct-Lite)
- `dataset_name`: Dataset to use for reasoning tasks (e.g., aqua)
- `method`: Reasoning method (cot, tot, mcts, l2m)
- `num_samples`: Number of reasoning traces to collect per example

## 📊 Supported Datasets

Support any (multiple) choice question data structured as follows:

```json
{
  "question": XXX,
  "options": ["A)XX", "B)XX", "C)XX"],
  "answer": "C"
}
```

## 🛠️ Creating Custom Datasets

You can create your own custom datasets to use with the Landscape of Thoughts framework. The framework supports multiple-choice question datasets in JSONL format.

For detailed instructions on creating, validating, and using custom datasets, see our [Custom Datasets Guide](lot/doc/custom_datasets.md).

## 🤖 Supported Models

All open-source models are accessible via API, either vllm, or API provider, as long as the log probability of each token is accessible.

## 📜 Citation

```bibtex
@article{zhou2025landscape,
  title={Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models},
  author={Zhou, Zhanke and Zhu, Zhaocheng and Li, Xuan and Galkin, Mikhail and Feng, Xiao and Koyejo, Sanmi and Tang, Jian and Han, Bo},
  journal={arXiv preprint arXiv:2503.22165},
  year={2025},
  url={https://arxiv.org/abs/2503.22165}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
