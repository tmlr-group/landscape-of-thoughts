#!/usr/bin/env python3
"""
Example script to demonstrate how to update the Together AI model information.

This script shows how to use the update_together_models.py script to fetch
the latest model information from the Together AI API.
"""

import os
import sys
import argparse

# Add the parent directory to the path so we can import the lot package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.update_together_models import get_together_models, save_model_info

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Update Together AI model information example")
    parser.add_argument("--api-key", help="Together AI API key")
    args = parser.parse_args()
    
    # Get the API key from the command line or environment
    api_key = args.api_key or os.environ.get("TOGETHERAI_API_KEY")
    if not api_key:
        print("Error: No API key provided. Please provide an API key using the --api-key argument")
        print("or set the TOGETHERAI_API_KEY environment variable.")
        return 1
    
    try:
        print("Fetching Together AI model information...")
        models = get_together_models(api_key)
        
        # Print some information about the models
        print(f"Found {len(models)} models available for inference")
        
        # Print the first 5 models
        if models:
            print("\nExample models:")
            for i, model in enumerate(models[:5]):
                print(f"  {i+1}. {model['model_name']} - {model['display_name']}")
            
            if len(models) > 5:
                print(f"  ... and {len(models) - 5} more.")
        
        # Save the model information
        output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  "models", "data", "togetherAI_model_infos.json")
        save_model_info(models, output_path)
        
        print(f"\nSuccessfully saved model information to {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 