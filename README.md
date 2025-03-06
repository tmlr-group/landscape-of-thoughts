<div align="center">

<h1>🌌 Landscape of Thoughts</h1>
<h3>Visual Reasoning Paths of LLMs through Dimensional Projection</h3>

[![Paper](https://img.shields.io/badge/arXiv-2311.03191-b31b1b)](https://arxiv.org/abs/2311.03191)
[![GitHub Stars](https://img.shields.io/github/stars/tmlr-group/DeepInception?style=social)](https://github.com/tmlr-group/DeepInception)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-blue)](https://huggingface.co/datasets/GazeEzio/Landscape-of-Thought)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13P51V3D6B89iZQjJ3Qaim2bBWOmTLoHe?usp=sharing)

|     ![demo](imgs/demo.png)     |
| :----------------------------: |
| Diagram of Lansacpe of Thought |

</div>

---

<!-- > [!TIP]
> Hello -->
<!-- > [!CAUTION]
> Hello -->

## 🏔️ Setup Environment

Before start analysing your own data, you may need to setup environment ([link](<(res/setup_environment.md)>)) and model ([link](res/setup_model.md)).

## 🔍 Analysis Pipeline

| Step                 | Command                                    | Output                                     | Documents                              |
| -------------------- | ------------------------------------------ | ------------------------------------------ | -------------------------------------- |
| 1. Collect Responses | `python step1-sample-reasoning-trace.py`   | `./exp-data-scale/*/thoughts/*.json`       | [link](./res/setup_environment.md#L43) |
| 2. Compute Distances | `python step-2-compute-distance-matrix.py` | `./exp-data-scale/*/distance_matrix/*.pkl` | [link](./res/setup_environment.md#L71) |
| 3. Generate Plots    | `python step_3_plot_landscape.py`          | `./figures/*.png`                          |                                        |

## 🚀 Quickstart Pipeline

```bash
# Full Workflow Example
export TOGETHERAI_API_KEY=your_key_here

# Step 1: Data Collection
python step_1_sample_reasoning_trace.py \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
  --dataset_name aqua

# Step 2: Distance Calculation
python step_2_compute_distance_matrix.py \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
  --dataset_name aqua

# Step 3: Visualization
python step_3_plot_landscape.py \
  --model_name Meta-Llama-3.1-8B-Instruct-Turbo \
  --dataset_name aqua
```

## 📜 Citation

```bibtex
@inprocessing{
  title={Landscape of Thought: XXX},
  author={XXX},
  booktitle={arXiv},
  year={2025},
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
