import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def compute_cosine_similarity(df):
    # Extract the sequences and their corresponding levels (Step #)
    df['level'] = df['sequence'].str.extract(r'Step (\d+)')
    
    # Group by level and calculate cosine similarity for each group
    similarity_results = {}
    
    for level, group in df.groupby('level'):
        # Use CountVectorizer to create a matrix of token counts
        count_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), lowercase=True)
        count_matrix = count_vectorizer.fit_transform(group['sequence'])
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(count_matrix)
        
        # Store results in a dictionary
        similarity_results[level] = cosine_sim
    
    return similarity_results


def levenshtein_distance(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize the boundary conditions
    for i in range(m + 1):
        dp[i][0] = i  # Deleting all characters from text1
    for j in range(n + 1):
        dp[0][j] = j  # Inserting all characters to text1 to form text2
        
    # Fill the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No operation needed
            else:
                dp[i][j] = min(dp[i - 1][j],    # Deletion
                               dp[i - 1][j - 1],  # Substitution
                               dp[i][j - 1]) + 1  # Insertion
                
    # Normalize the distance to the range [0, 1]
    return 1 - dp[m][n] / max(m, n) if max(m, n) > 0 else 1.0  # Handle case where both strings are empty


def hamming_distance(text1, text2):
    if len(text1) != len(text2):
        raise ValueError("Strings must be equal length")
    return sum(c1 != c2 for c1, c2 in zip(text1, text2)) / len(text1)

def word2vec_similarity(texts, vector_size=100):
    # Train Word2Vec model
    sentences = [text.split() for text in texts]
    model = Word2Vec(sentences, vector_size=vector_size, min_count=1)
    
    def text_to_vector(text):
        words = text.split()
        vectors = [model.wv[word] for word in words if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)
    
    # Convert texts to vectors
    vectors = [text_to_vector(text) for text in texts]
    
    # Calculate pairwise cosine similarity
    similarity_matrix = np.zeros((len(texts), len(texts)))
    for i in range(len(texts)):
        for j in range(len(texts)):
            v1, v2 = vectors[i], vectors[j]
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:  # Avoid division by zero
                similarity_matrix[i][j] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            else:
                similarity_matrix[i][j] = 0.0  # If either vector is zero, similarity is zero
            
    return similarity_matrix

def compute_similarity(df, method):
    """Compute similarities (or distances) based on the specified method."""
    
    results = {}
    for level, group in df.groupby('level'):
        sequences = group['distance'].tolist()
        
        if method == 'euclidean':
            # Compute Euclidean distance matrix
            distance_matrix = np.zeros((len(sequences), len(sequences)))
            for i in range(len(sequences)):
                for j in range(len(sequences)):
                    distance_matrix[i][j] = np.linalg.norm(sequences[i] - sequences[j])
            results[level] = distance_matrix
            
        elif method == 'manhattan':
            # Compute Manhattan distance matrix
            distance_matrix = np.zeros((len(sequences), len(sequences)))
            for i in range(len(sequences)):
                for j in range(len(sequences)):
                    distance_matrix[i][j] = np.sum(np.abs(sequences[i] - sequences[j]))
            results[level] = distance_matrix
            
        elif method == 'cosine':
            # Compute cosine similarity matrix
            distance_matrix = np.zeros((len(sequences), len(sequences)))
            for i in range(len(sequences)):
                for j in range(len(sequences)):
                    dot_product = np.dot(sequences[i], sequences[j])
                    norm_i = np.linalg.norm(sequences[i])
                    norm_j = np.linalg.norm(sequences[j])
                    distance_matrix[i][j] = dot_product / (norm_i * norm_j)
            results[level] = distance_matrix

    return results

def get_similarity_all_samples(all_df_thoughts, methods=['levenshtein', 'cosine', 'word2vec']):
    all_means = {method:{} for method in methods}
    for df_thoughts in tqdm(all_df_thoughts): # all sample 
        for method in methods: # all methods
            # print(f"==> Computing similarity for {method}")
            raw_matrix = compute_similarity(df_thoughts, method=method)
            for level, means in raw_matrix.items(): # all level
                np.fill_diagonal(means, np.nan)
                if not np.isnan(means).all():
                    mean_val = np.nanmean(means)
                    if level not in all_means[method]:
                        all_means[method][level] = []
                    all_means[method][level].append(mean_val)
    return all_means