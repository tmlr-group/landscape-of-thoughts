# Trace of Thoughts Sampling

This repository contains scripts for sampling reasoning traces from language models using different reasoning methods (CoT, ToT, MCTS) on various datasets.

## Scripts

### 1. `sample_reasoning_trace.py`

This script is a direct implementation of the sampling logic from the original `step_1_sample_reasoning_trace.py` but using the `lot` library structure. It provides detailed control over the sampling process.

```bash
python sample_reasoning_trace.py \
  --model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" \
  --dataset_name="aqua" \
  --method="cot" \
  --samples=10 \
  --start_index=0 \
  --end_index=50 \
  --data_path="data/aqua.jsonl"
```

### 2. `sample_with_lot.py`

This script provides a higher-level interface using the `lot` library, with a `sample` function that can be used with any dataset, model, and algorithm.

```bash
python sample_with_lot.py \
  --model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" \
  --port=8000 \
  --dataset_name="aqua" \
  --data_path="data/aqua.jsonl" \
  --method="cot" \
  --num_samples=10 \
  --start_index=0 \
  --end_index=50 \
  --prompt_file="prompts/aqua_cot.txt"
```

### 3. `simple_sample.py`

This script is a simplified version that closely follows the example in the suggestion.md file, demonstrating how to use the `lot` library with minimal code.

```bash
python simple_sample.py \
  --model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" \
  --dataset_name="aqua" \
  --data_path="data/aqua.jsonl" \
  --algorithm_name="cot" \
  --num_samples=10 \
  --prompt_file="prompts/aqua_cot.txt"
```

## Supported Datasets

- MMLU
- AQuA
- CommonsenseQA
- StrategyQA
- Custom JSON/JSONL datasets

## Supported Algorithms

- Standard (direct answering)
- Chain of Thought (CoT)
- Tree of Thoughts (ToT)
- Monte Carlo Tree Search (MCTS)

## Output Format

The scripts save the sampling results in JSON files with the following structure:

```json
{
  "dataset": "aqua",
  "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
  "method": "cot",
  "model_input": "...",
  "answers": ["Answer is: A. ...", "Answer is: B. ...", ...],
  "answer_gt_full": "Answer is: A. ...",
  "answer_gt_short": "A",
  "answer_gt_expl": "...",
  "trial_thoughts": [
    [["thought-1", "thought-2", ...], "A", true],
    [["thought-1", "thought-2", ...], "B", false],
    ...
  ],
  "accuracy": 0.7
}
```

For ToT and MCTS methods, the scripts also save the search trees in JSON format.

## Requirements

- Python 3.8+
- lot library
- fire
- models.opensource_API (for model access)
