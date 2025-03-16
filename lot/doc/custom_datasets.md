# Creating Custom Datasets for Landscape of Thoughts

This guide explains how to create, validate, and use custom datasets with the Landscape of Thoughts (LoT) framework.

## Dataset Format

The LoT framework supports multiple-choice question datasets in JSONL format. Each line in the file should be a JSON object with the following structure:

```json
{
  "question": "What is the capital of France?",
  "options": ["A) London", "B) Berlin", "C) Paris", "D) Madrid"],
  "answer": "C"
}
```

The required fields are:

- `question`: The question text
- `options`: An array of answer options (prefixed with "A)", "B)", etc.)
- `answer`: The correct answer letter (A, B, C, etc.)

## Step-by-Step Guide

### 1. Creating a Custom Dataset

Create a JSONL file (e.g., `my_dataset.jsonl`) with your questions:

```bash
echo '{"question": "What is the capital of France?", "options": ["A) London", "B) Berlin", "C) Paris", "D) Madrid"], "answer": "C"}' > data/my_dataset.jsonl
echo '{"question": "Which planet is known as the Red Planet?", "options": ["A) Venus", "B) Mars", "C) Jupiter", "D) Saturn"], "answer": "B"}' >> data/my_dataset.jsonl
echo '{"question": "What is 2+2?", "options": ["A) 3", "B) 4", "C) 5", "D) 6"], "answer": "B"}' >> data/my_dataset.jsonl
```

### 2. Verifying Your Dataset

Use this Python script to verify your dataset:

```python
import json

# Load and check the dataset
dataset_path = "data/my_dataset.jsonl"
with open(dataset_path, 'r') as f:
    dataset = [json.loads(line) for line in f]

# Print dataset statistics
print(f"Dataset loaded: {len(dataset)} examples")
for i, example in enumerate(dataset):
    print(f"Example {i}:")
    print(f"  Question: {example['question']}")
    print(f"  Options: {example['options']}")
    print(f"  Answer: {example['answer']}")
    # Verify the answer corresponds to a valid option
    answer_idx = ord(example['answer']) - ord('A')
    if 0 <= answer_idx < len(example['options']):
        print(f"  Answer text: {example['options'][answer_idx]}")
    else:
        print(f"  WARNING: Answer '{example['answer']}' does not match any option!")
    print()
```

### 3. Validating Dataset Format

To ensure your dataset works correctly with the framework, check for these common issues:

1. **Valid JSON format**: Each line must be a valid JSON object
2. **Required fields**: Each example must have `question`, `options`, and `answer` fields
3. **Answer format**: The `answer` field should be a single letter (A, B, C, etc.)
4. **Options format**: The `options` field should be a list of strings
5. **Consistency**: The answer letter should correspond to a valid option index

Here's a validation script that checks for these issues:

```python
import json
import re

def validate_dataset(dataset_path):
    """Validate a dataset for use with the LoT framework."""
    try:
        with open(dataset_path, 'r') as f:
            lines = f.readlines()

        print(f"Checking {len(lines)} examples...")
        valid_examples = 0
        issues = []

        for i, line in enumerate(lines):
            line_num = i + 1
            try:
                # Check JSON parsing
                example = json.loads(line.strip())

                # Check required fields
                for field in ['question', 'options', 'answer']:
                    if field not in example:
                        issues.append(f"Line {line_num}: Missing required field '{field}'")
                        continue

                # Check options format
                if not isinstance(example['options'], list):
                    issues.append(f"Line {line_num}: 'options' must be a list")
                    continue

                if len(example['options']) == 0:
                    issues.append(f"Line {line_num}: 'options' list is empty")
                    continue

                # Check answer format
                if not re.match(r'^[A-E]$', example['answer']):
                    issues.append(f"Line {line_num}: 'answer' must be a single letter (A-E)")
                    continue

                # Check answer validity
                answer_idx = ord(example['answer']) - ord('A')
                if answer_idx < 0 or answer_idx >= len(example['options']):
                    issues.append(f"Line {line_num}: Answer '{example['answer']}' does not match any option")
                    continue

                valid_examples += 1

            except json.JSONDecodeError:
                issues.append(f"Line {line_num}: Invalid JSON format")

        # Print results
        print(f"Validation complete: {valid_examples}/{len(lines)} examples are valid")
        if issues:
            print("\nIssues found:")
            for issue in issues:
                print(f"- {issue}")
        else:
            print("\nNo issues found! Your dataset is ready to use.")

        return valid_examples == len(lines)

    except FileNotFoundError:
        print(f"Error: File not found: {dataset_path}")
        return False

# Example usage
validate_dataset("data/my_dataset.jsonl")
```

### 4. Using Your Custom Dataset

Once your dataset is ready, you can use it with the LoT framework:

```bash
python main.py \
  --task all \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct-Lite \
  --dataset_name my_dataset \
  --data_path data/my_dataset.jsonl \
  --method cot \
  --num_samples 10 \
  --start_index 0 \
  --end_index 3
```

You can also use the Python API directly:

```python
from lot import sample, calculate, plot

# Sample reasoning traces
features, metrics = sample(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    dataset_name="my_dataset",
    data_path="data/my_dataset.jsonl",
    method="cot",
    num_samples=10,
    start_index=0,
    end_index=3
)

# Calculate distance matrices
distance_matrices = calculate(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    dataset_name="my_dataset",
    data_path="data/my_dataset.jsonl",
    method="cot",
    start_index=0,
    end_index=3
)

# Generate visualizations
plot(
    model_name="Meta-Llama-3-8B-Instruct-Lite",
    dataset_name="my_dataset",
    method="cot",
    plot_type="method"
)
```

## Advanced Usage

### Registering a Custom Dataset

For more advanced usage, you can register your custom dataset with the framework by creating a dataset class:

1. Create a new Python file in the `lot/datasets` directory:

```python
# lot/datasets/my_custom_dataset.py
from .base_dataset import BaseDataset

class MyCustomDataset(BaseDataset):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.dataset_name = "my_custom_dataset"
        self.answer_pattern = r"Answer is: ([A-E])"  # Regex pattern to extract answers

    def get_query(self, idx):
        """Format the query for the model."""
        example = self[idx]
        question = example["question"]
        options = "\n".join(example["options"])
        return f"{question}\n\n{options}"

    def get_prompt(self, method, default_prompt):
        """Return a method-specific prompt template."""
        if method == "cot":
            return "Please solve the following multiple-choice question step-by-step:\n\n{question}\n\nThink through this carefully:"
        return default_prompt
```

2. Register your dataset in `lot/datasets/__init__.py`:

```python
# lot/datasets/__init__.py
from .my_custom_dataset import MyCustomDataset

# Add to the dataset registry
DATASET_REGISTRY = {
    # ... existing datasets ...
    "my_custom_dataset": MyCustomDataset,
}
```

3. Use your registered dataset:

```bash
python main.py \
  --task all \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct-Lite \
  --dataset_name my_custom_dataset \
  --data_path data/my_custom_dataset.jsonl \
  --method cot
```

This approach gives you more control over how your dataset is processed, including custom prompts and answer extraction patterns.

## Troubleshooting

If you encounter issues when using your custom dataset with the framework, check these common problems:

1. **Dataset not found**: Ensure the path to your dataset file is correct and the file exists

   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'data/my_dataset.jsonl'
   ```

   Solution: Double-check the file path and create the directory if needed

2. **JSON parsing errors**: Make sure each line in your JSONL file is a valid JSON object

   ```
   json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes
   ```

   Solution: Validate your JSON format using the validation script provided above

3. **Missing answer pattern**: If answers aren't being extracted correctly

   ```
   Warning: Could not extract answer from response
   ```

   Solution: When registering a custom dataset, ensure the `answer_pattern` regex matches the model's output format

4. **Low accuracy**: If your model is performing poorly on your dataset

   ```
   Average accuracy: 0.1250
   ```

   Solution:

   - Check that questions are clear and unambiguous
   - Try a different reasoning method (e.g., switch from 'cot' to 'tot')
   - Ensure options are properly formatted with clear distinctions
   - Consider using a more capable model

5. **Memory issues**: If processing large datasets
   ```
   MemoryError: ...
   ```
   Solution: Process your dataset in smaller batches by using the `start_index` and `end_index` parameters

For more complex issues, you can examine the generated files in the `save_root` directory (default: "exp-data") to debug the model's reasoning process.

## Tips for Better Results

- Ensure your dataset follows the required format with `question`, `options`, and `answer` fields
- The `answer` field should contain the letter corresponding to the correct option (e.g., "A", "B", "C", etc.)
- For best results, make sure your questions are clear and unambiguous
- You can include additional fields (e.g., explanations, categories) that won't affect the core functionality
- Consider the formatting of options - including the letter prefix (e.g., "A)") helps models identify options
- Test your dataset with different reasoning methods to find what works best for your questions
