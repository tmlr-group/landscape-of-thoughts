import json
import os

import pandas as pd
import pytest
import requests
from litellm import completion


# testing the calling function without calling API
def test_completion_openai():
    try:
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role":"user", "content":"Why is LiteLLM amazing?"}],
            mock_response="LiteLLM is awesome"
        )
        # Add any assertions here to check the response
        print(response)
        assert(response['choices'][0]['message']['content'] == "LiteLLM is awesome")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
        
# Get provided model information, return in dataframe or JSON
def get_all_model_infos(api, return_df=False):
    url = "https://api.together.xyz/v1/models"

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer "+api
    }

    response = requests.get(url, headers=headers)

    model_infos = []
    for model_info in response.json():
        model_name = model_info['id']
        context_length = model_info.get('context_length', 'Unknown')
        in_price = model_info['pricing']['input']
        out_price = model_info['pricing']['output']

        model_infos.append(
            {"model_name": model_name, "context_length": context_length, "in_price": in_price, "out_price": out_price}
        )
    if return_df:
        model_infos = pd.DataFrame(model_infos).T.reset_index()
        model_infos.columns = ['ID', 'context_length', 'in_price', 'out_price']

        
    return model_infos

def get_model_infos(model_name, api, return_df=False):
    url = "https://api.together.xyz/v1/models"

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer "+api
    }
    
    response = requests.get(url, headers=headers)

    model_infos = []
    for model_info in response.json():
        if model_name.lower() in model_info['id']:
            model_infos.append(
                '''ID: {id}, context_length: {cl}, in_price: {in_p}, out_price: {out_p}'''.format(
                    id=model_info['id'], cl=model_info.get('context_length', 'Unknown'), 
                    in_p=model_info['pricing']['input'], out_p=model_info['pricing']['output'],
                )
            )

    if return_df:
        import re

        # Parse the data
        parsed_data = []
        for entry in model_infos:
            parts = re.split(r'[:,]', entry)
            parsed_data.append({
                "ID": parts[1].strip(),
                "context_length": int(parts[3].strip()),
                "in_price / M": float(parts[5].strip()),
                "out_price / M": float(parts[7].strip())
            })

        # Create DataFrame
        model_infos = pd.DataFrame(parsed_data)
    
    return model_infos

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_api_key():
    try:
        api_key = os.environ['TOGETHERAI_API_KEY']
        return api_key
    except KeyError:
        print("API key not found. Please export the API key using 'export TOGETHERAI_API_KEY=your_api_key' in your terminal.")
        return None