# Building and Testing Wheels for Landscape of Thoughts

This guide explains how to build and test wheel distributions for the Landscape of Thoughts package.

## Prerequisites

Make sure you have the following tools installed:

```bash
pip install build wheel twine pytest
```

## Building the Wheel

To build the wheel distribution, run:

```bash
python -m build
```

This will create both source distribution (`.tar.gz`) and wheel distribution (`.whl`) files in the `dist/` directory.

## Testing the Wheel Locally

### Method 1: Install from Local Wheel

```bash
# Install the wheel (replace X.X.X with the actual version)
conda create -n lot_pip python=3.10
conda activate lot_pip
pip install dist/landscape_of_thoughts-0.1.0-py3-none-any.whl

# Verify installation
pip list | grep landscape-of-thoughts

# Test the command-line interface
lot --help
```

### Method 2: Using a Virtual Environment

```bash
# Create a new virtual environment
python -m venv test_venv

# Activate the virtual environment
# On Windows:
test_venv\Scripts\activate
# On Unix or MacOS:
source test_venv/bin/activate

# Install the wheel
pip install dist/landscape_of_thoughts-0.1.0-py3-none-any.whl

# Test the command-line interface
lot --help

# Run a minimal example
lot --task all --model_name meta-llama/Llama-3.2-1B-Instruct --dataset_name aqua --method cot --num_samples 10 --start_index 0 --end_index 5 --plot_type method --output_dir figures/landscape --local --local_api_key token-abc123


# Deactivate when done
deactivate
```

### Method 3: Install in Development Mode

For developers who want to make changes and test them immediately:

```bash
# Install in development mode
pip install -e .

# Now changes to the code will be immediately available
# without needing to reinstall
```

## Validating the Wheel

You can check if your wheel is valid with `twine`:

```bash
twine check dist/*
```

## Testing Installation from PyPI Test Server (Optional)

If you want to test the full PyPI workflow without publishing to the actual PyPI:

```bash
# Upload to TestPyPI
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Then in a new virtual environment
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple landscape-of-thoughts
```

## Troubleshooting

1. **Missing files in the wheel**: Check your `MANIFEST.in` and `setup.py` to ensure all required files are included.

2. **Dependency issues**: If dependencies aren't resolving correctly, try installing with the `--no-deps` flag and then manually installing dependencies.

3. **Entry point not working**: Verify that `main.py` is included in the wheel and that the entry point configuration in `setup.py` is correct.

4. **Import errors after installation**: Check that all necessary packages are listed in the `install_requires` in `setup.py`.

## Running Tests

If you have tests for your package, you can run them after installing the wheel:

```bash
# If you have pytest tests
pytest

# Or a specific test
pytest tests/test_specific.py
```

Remember to check that imports in your tests use the installed package path, not relative imports, when testing the installed wheel.
