import pandas as pd

from transformers import AutoTokenizer

import matplotlib.pyplot as plt

import numpy as np

EXTERNAL_DATASET_PATH = "updated_data.csv"
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

extra_data = pd.read_csv(EXTERNAL_DATASET_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)



def Extract_Influence_token(data_frame):

    tokens = []
    influence_scores = []

    for smiles, influence_score in zip(data_frame['SMILES'], data_frame['influence_score']):
        token = tokenizer(smiles, padding=True, truncation=True, return_tensors="pt")
        influence_scores.append(influence_score)
        tokens.append(token['input_ids'][0].tolist())

    return tokens, influence_scores

def pad_tokens(tokens):

    for i, token in enumerate(tokens):
        if len(token) < 49:
            padding_needed = 49 - len(token)

            tokens[i] = token + [0] * padding_needed 
    
    return tokens

def filtered_dataset(dataframe):

    _, influence_score = Extract_Influence_token(dataframe)

    mean = np.mean(influence_score)
    std = np.std(influence_score)

    upper_bound = mean + std
    lower_bound = mean - std

    filtered_df = dataframe[(influence_score >= lower_bound) & (influence_score <= upper_bound)]

    filtered_df.to_csv("filtered_extrapoints.csv", index = False)

    return filtered_df

def plot_influence_scatter(influence_scores):

    indices = range(len(influence_scores))

    plt.figure(figsize=(8, 5))
    plt.scatter(indices, influence_scores, color='b', alpha=0.6, label='Influence Score')

    plt.axhline(y= np.mean(influence_scores), color='r', linestyle='--', linewidth=1)  
    plt.axhline(y= np.mean(influence_scores) + np.std(influence_scores), color='r', linestyle='--', linewidth=1)
    plt.axhline(y= np.mean(influence_scores) - np.std(influence_scores), color='r', linestyle='--', linewidth=1)
    plt.xlabel('Data Point Index')
    plt.ylabel('Influence Score')
    plt.title('Influence Score Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()


filtered_dataset(extra_data)
