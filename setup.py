#!/usr/bin/env python
import os
from setuptools import setup, find_packages

# Use the same version as in pyproject.toml
version = "0.1.0"

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="landscape-of-thoughts",
    version=version,
    description="Visualizing the Reasoning Process of Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Zhanke Zhou, Zhaocheng Zhu, Xuan Li, Mikhail Galkin, Xiao Feng, Sanmi Koyejo, Jian Tang, Bo Han",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["llm", "reasoning", "visualization", "landscape", "thoughts"],
    packages=find_packages(),
    package_data={"lot": ["**/*.jsonl"]},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "aiohttp>=3.11.13",
        "aiosignal>=1.3.2",
        "anyio>=4.8.0",
        "async-timeout>=5.0.1",
        "attrs>=25.1.0",
        "certifi>=2025.1.31",
        "charset-normalizer>=3.4.1",
        "datasets>=3.3.2",
        "fastapi>=0.115.11",
        "filelock>=3.17.0",
        "fsspec>=2024.12.0",
        "huggingface-hub>=0.29.2",
        "httpx>=0.28.1",
        "joblib>=1.4.2",
        "numpy>=2.1.3",
        "openai>=1.65.5",
        "pandas>=2.2.3",
        "pillow>=11.1.0",
        "pydantic>=2.10.6",
        "python-dotenv>=1.0.1",
        "pyyaml>=6.0.2",
        "requests>=2.32.3",
        "scikit-learn>=1.6.1",
        "scipy>=1.15.2",
        "tokenizers>=0.21.0",
        "tqdm>=4.67.1",
        "transformers>=4.49.0",
        "typer>=0.15.2",
        "urllib3>=2.3.0",
        "uvicorn>=0.34.0",
        "plotly>=5.24.0",
        "fire>=0.7.0",
        "sympy>=1.13.1",
        "pylatexenc>=2.10.0",
        "matplotlib>=3.10.0",
        "kaleido>=0.2.1",
        "together>=1.5.5",
        "nbformat>=4.2.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.10.0",
            "flake8>=6.0.0"
        ],
    },
    py_modules=["main"],
    entry_points={
        "console_scripts": [
            "lot=lot.main:main",
        ],
    },
    project_urls={
        "Homepage": "https://github.com/tmlr-group/landscape-of-thoughts",
        "Bug Tracker": "https://github.com/tmlr-group/landscape-of-thoughts/issues",
        "Documentation": "https://github.com/tmlr-group/landscape-of-thoughts",
        "Paper": "https://arxiv.org/abs/2503.22165",
    },
) 