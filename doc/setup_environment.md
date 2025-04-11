# 🔧 Preparation

## 🤖 Setting up Model

We recommand using TogetherAI API to get access to the model, or using vllm to host your own model, see [here](res/setup_model.md).

## 📊 Preparaing Data

### Using Pre-computed Data

We provide the exact same data used for visulizing the plot in our paper, you can use the following command to download the data.

```bash
git lfs clone git@hf.co:datasets/GazeEzio/Landscape-Data
```

The expected file tree is:

```
Landscape-of-Thoughts/
│    ...
├──  Landscape-Data/
│    ├── aqua
│    │   ├── distance_matrix
│    │   └── thoughts
│    └──...
│   ...
```

### Using Your Own Data

Alternatively, you can use your own dataset for obtaining the visulization data.
In what follows, we provide an example of how to obtain the data for `AQuA` dataset with `CoT` using TogetherAI. The employed dataset can be found in [data](../data).

- Step 1. Obtain Responses from LLM

```bash
export TOGETHERAI_API_KEY=XXX

python step1-sample-reasoning-trace.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name aqua --dataset_path data/aqua.jsonl --method cot
```

The results will be saved as `./exp-data-scale/aqua/thoughts/Meta-Llama-3.1-8B-Instruct-Turbo--cot--aqua--*.json`.

Expected file tree:

```
exp-data-scale
├── aqua <== dataset_name
│   ├── distance_matrix <== the data for ploting
│   └── thoughts <== the LLM raw responses
├── commonsenseqa
│   ├── distance_matrix
│   └── thoughts
├── mmlu
│   ├── distance_matrix
│   └── thoughts
└── strategyqa
    ├── distance_matrix
    └── thoughts
```

- Step 2. Calculating the Distance Matrix

We need to map the natural language to 2D landscape by calculating the respective distance matrix. To do this, run:

```bash
python step-2-compute-distance-matrix.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name aqua --dataset_path data/aqua.jsonl --method cot
```

The results will be saved as `./exp-data-scale/aqua/distance_matrix/Meta-Llama-3.1-8B-Instruct-Turbo--cot--aqua--*.pkl`, which is ready for plotting.
