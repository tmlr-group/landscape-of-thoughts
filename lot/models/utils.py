#!/usr/bin/env python3
"""
Utility functions for working with language model APIs.

This module provides functions for retrieving API keys, fetching model information,
and managing model data for various API providers like Together AI.
"""
import json
import os
import time
from typing import Dict, List, Any, Optional, Union, Literal

import pandas as pd
import requests

# Constants
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TOGETHER_MODEL_INFO_PATH = os.path.join(DEFAULT_CACHE_DIR, "togetherAI_model_infos.json")
CACHE_EXPIRY = 7 * 24 * 60 * 60  # 7 days in seconds

def get_api_key(provider: str = "together") -> Optional[str]:
    """Get API key for the specified provider from environment variables.
    
    Args:
        provider: The API provider name. Options: "together", "openai". Default: "together".
        
    Returns:
        The API key if found, None otherwise.
        
    Raises:
        ValueError: If an invalid provider is specified.
    """
    if provider.lower() == "together":
        env_var = "TOGETHERAI_API_KEY"
    elif provider.lower() == "openai":
        env_var = "OPENAI_API_KEY"
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: together, openai")
    
    api_key = os.environ.get(env_var)
    if not api_key:
        print(f"{provider.capitalize()} API key not found. Please set the {env_var} environment variable.")
    
    return api_key

def get_together_models(api_key: Optional[str] = None, force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Fetch the list of available models from the Together AI API with caching.
    
    Args:
        api_key: Together AI API key. If not provided, will try to get from environment.
        force_refresh: If True, ignore cached data and fetch fresh data from API.
        
    Returns:
        List of model information dictionaries.
        
    Raises:
        ValueError: If API key is not provided and not found in environment.
        RuntimeError: If the API request fails.
    """
    # Check if we can use cached data
    if not force_refresh and os.path.exists(TOGETHER_MODEL_INFO_PATH):
        # Check if cache is still valid (less than CACHE_EXPIRY seconds old)
        cache_age = time.time() - os.path.getmtime(TOGETHER_MODEL_INFO_PATH)
        if cache_age < CACHE_EXPIRY:
            try:
                with open(TOGETHER_MODEL_INFO_PATH, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # If cache is corrupted, continue to fetch fresh data
                pass
    
    # Get API key from environment if not provided
    if api_key is None:
        api_key = get_api_key("together")
        if not api_key:
            raise ValueError(
                "API key not provided and TOGETHERAI_API_KEY environment variable not set.\n"
                "Please set the TOGETHERAI_API_KEY environment variable using export TOGETHERAI_API_KEY=<your_api_key>"
            )
    
    # Set up the API request
    url = "https://api.together.xyz/v1/models"
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_key}"
    }
    
    # Make the request
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch models from Together AI API: {e}")
    
    # Parse the response
    data = response.json()
    
    # Extract and format the model information
    models = []
    for model in data:
        models.append({
            "model_name": model.get("id"),
            "display_name": model.get("display_name", model.get("id")),
            "context_length": model.get("context_length", 4096),
            "description": model.get("description", ""),
            "pricing": {
                "input": model.get("pricing", {}).get("input", 0),
                "output": model.get("pricing", {}).get("output", 0)
            },
            "capabilities": model.get("capabilities", [])
        })
    
    # Save to cache
    os.makedirs(os.path.dirname(TOGETHER_MODEL_INFO_PATH), exist_ok=True)
    try:
        with open(TOGETHER_MODEL_INFO_PATH, 'w') as f:
            json.dump(models, f, indent=2)
    except IOError as e:
        print(f"Warning: Failed to cache model information: {e}")
    
    return models

def get_model_info(model_name: str, api_key: Optional[str] = None, 
                  format: Literal["dict", "df", "str"] = "dict") -> Union[Dict, pd.DataFrame, str]:
    """
    Get information for a specific model from Together AI.
    
    Args:
        model_name: Name or partial name of the model to search for.
        api_key: Together AI API key. If not provided, will try to get from environment.
        format: Output format - "dict", "df" (DataFrame), or "str".
        
    Returns:
        Model information in the requested format.
    """
    models = get_together_models(api_key)
    
    # Filter models by name (case-insensitive partial match)
    model_name_lower = model_name.lower()
    matching_models = [
        model for model in models 
        if model_name_lower in model["model_name"].lower()
    ]
    
    if not matching_models:
        return {} if format == "dict" else pd.DataFrame() if format == "df" else ""
    
    if format == "dict":
        return matching_models
    
    if format == "str":
        result = []
        for model in matching_models:
            result.append(
                f'''ID: {model["model_name"]}, context_length: {model["context_length"]}, '''
                f'''in_price: {model["pricing"]["input"]}, out_price: {model["pricing"]["output"]}'''
            )
        return "\n".join(result)
    
    # DataFrame format
    df_data = []
    for model in matching_models:
        df_data.append({
            "ID": model["model_name"],
            "context_length": model["context_length"],
            "in_price / M": float(model["pricing"]["input"]),
            "out_price / M": float(model["pricing"]["output"])
        })
    
    return pd.DataFrame(df_data)

def save_model_info(output_path: Optional[str] = None) -> None:
    """
    Save the model information to a JSON file.
    
    Args:
        output_path: Path to save the JSON file. If None, uses the default path.
        
    Raises:
        IOError: If the file cannot be written.
    """
    if output_path is None:
        output_path = TOGETHER_MODEL_INFO_PATH
    
    # Fetch fresh model data
    models = get_together_models(force_refresh=True)
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the model information to the file
    try:
        with open(output_path, 'w') as f:
            json.dump(models, f, indent=2)
        print(f"Successfully saved model information to {output_path}")
        print(f"Found {len(models)} models available for inference")
    except IOError as e:
        raise IOError(f"Failed to write model information to {output_path}: {e}")

# For backward compatibility
def get_all_model_infos(api_key, return_df=False):
    """Legacy function for backward compatibility."""
    models = get_together_models(api_key)
    
    if not return_df:
        return models
    
    df_data = []
    for model in models:
        df_data.append({
            "model_name": model["model_name"],
            "context_length": model["context_length"],
            "in_price": model["pricing"]["input"],
            "out_price": model["pricing"]["output"]
        })
    
    df = pd.DataFrame(df_data).T.reset_index()
    df.columns = ['ID', 'context_length', 'in_price', 'out_price']
    return df

# For backward compatibility
def get_model_infos(model_name, api_key, return_df=False):
    """Legacy function for backward compatibility."""
    format_type = "df" if return_df else "str"
    return get_model_info(model_name, api_key, format=format_type)
