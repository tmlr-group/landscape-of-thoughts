import numpy as np
from typing import Dict, List, Any, Tuple, Union, Optional
from tqdm import tqdm

from .models.base import BaseModel
from .datasets.base import BaseDataset
from .algorithms.base import BaseAlgorithm

def sample(dataset: BaseDataset, model: BaseModel, algorithm: BaseAlgorithm, 
           num_sample: int = 10) -> Tuple[List, Dict[str, List]]:
    """
    Sample reasoning paths from a dataset using a model and an algorithm.
    
    Args:
        dataset: The dataset to sample from.
        model: The model to use for sampling.
        algorithm: The algorithm to use for sampling.
        num_sample: Number of samples per query.
        
    Returns:
        Tuple[List, Dict[str, List]]: A tuple containing:
            - features: A list of features for each reasoning path.
            - metrics: A dictionary of metrics for each reasoning path.
    """
    features = []
    metrics = {
        'accuracy': [],
        'consistency': [],
        'uncertainty': [],
        'perplexity': []
    }
    
    # Sample reasoning paths for each query in the dataset
    for idx in tqdm(range(len(dataset)), desc="Sampling reasoning paths"):
        example = dataset[idx]
        query = example.get('query', '')
        answer = example.get('answer', '')
        
        # Sample multiple reasoning paths for the same query
        for _ in range(num_sample):
            # Run the algorithm on the query
            reasoning_steps = algorithm.run(model, query)
            
            # Extract features from the reasoning steps
            path_features = []
            for step in reasoning_steps:
                # Use the state as a feature
                state = step.get('state', '')
                
                # Calculate the likelihood of the state
                likelihood = model.get_likelihood(query, state)
                
                # Create a feature vector for the state
                feature = np.array([likelihood])
                
                # Add the feature to the path
                path_features.append(feature)
            
            # Add the path features to the features list
            features.append(path_features)
            
            # Calculate metrics for the path
            final_state = reasoning_steps[-1].get('state', '')
            
            # Calculate accuracy
            accuracy = 1.0 if answer in final_state else 0.0
            metrics['accuracy'].append(accuracy)
            
            # Calculate consistency
            consistency = 1.0 if len(set(step.get('thought', '') for step in reasoning_steps)) == len(reasoning_steps) else 0.0
            metrics['consistency'].append(consistency)
            
            # Calculate uncertainty
            uncertainty = 1.0 - model.get_likelihood(query, final_state)
            metrics['uncertainty'].append(uncertainty)
            
            # Calculate perplexity
            perplexity = -np.log(model.get_likelihood(query, final_state))
            metrics['perplexity'].append(perplexity)
    
    return features, metrics
