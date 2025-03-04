<div align="center">
<!-- [![Paper](https://img.shields.io/badge/arXiv-2311.03191-b31b1b)](https://arxiv.org/abs/2311.03191)
[![GitHub Stars](https://img.shields.io/github/stars/tmlr-group/DeepInception?style=social)](https://github.com/tmlr-group/DeepInception) -->

<h1>🌌 Landscape of Thoughts</h1>
<h3>Visual Reasoning Paths of LLMs through Dimensional Projection</h3>

[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-blue)](https://huggingface.co/datasets/GazeEzio/Landscape-of-Thought)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](TBD:ColabLink)

![demo](imgs/demo.png)

</div>

---

### 🔧 Installation Toolkit

```bash
# Create environment
conda create -n landscape python=3.10
pip3 install -r requirements.txt
```

For model, we recommand using TogetherAI API to get access to the model, you can follow the instruction in [here](). You can also use vllm to host your own model, see [here]().

### 📊 Preparaing Data

```bash
# Download Precomputed Data
git lfs clone git@hf.co:datasets/GazeEzio/Landscape-Data
```

```
Landscape-Data/
├── aqua
│   ├── distance_matrix
│   └── thoughts
└──...
```

### 🔍 Analysis Pipeline

| Step                 | Command                                    | Output                                     |
| -------------------- | ------------------------------------------ | ------------------------------------------ |
| 1. Collect Responses | `python step1-sample-reasoning-trace.py`   | `./exp-data-scale/*/thoughts/*.json`       |
| 2. Compute Distances | `python step-2-compute-distance-matrix.py` | `./exp-data-scale/*/distance_matrix/*.pkl` |
| 3. Generate Plots    | `python PLOT-landscape.py`                 | `./figures/*.png`                          |

## 🚀 Quickstart Pipeline

```bash
# Full Workflow Example
export TOGETHERAI_API_KEY=your_key_here

# Step 1: Data Collection
python step1-sample-reasoning-trace.py \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
  --dataset_name aqua

# Step 2: Distance Calculation
python step-2-compute-distance-matrix.py \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
  --dataset_name aqua

# Step 3: Visualization
python PLOT-landscape.py \
  --model_name Meta-Llama-3.1-8B-Instruct-Turbo \
  --dataset_name aqua
```

## 📚 Data Structure

## 📜 Citation

```bibtex

```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
