import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.utils import shuffle
from src.models import MoLFormerWithRegressionHeadMLM
from src.utils import SMILESDataset, SMILESextra, merge_datasets, loss_fig

# Define model names
TARGET_MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"  # Target model
PROXY_MODEL = Ridge()  # Simple proxy regression model for valuation

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_external_data(file_path="datasets/External-Dataset_for_Task2.csv"):
    """Loads the external dataset from a CSV file into a pandas DataFrame."""
    df = pd.read_csv(file_path)
    return df

def extract_representations(df, model_name=TARGET_MODEL_NAME):
    """Extracts embeddings from the penultimate layer of the target model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base_model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)  
    model = MoLFormerWithRegressionHeadMLM(base_model).to(device)
    model.eval()
    
    def tokenize(smiles):
        return tokenizer(smiles, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    
    embeddings = []
    with torch.no_grad():
        for smiles in tqdm(df["SMILES"], desc="Extracting representations"):
            tokens = tokenize(smiles)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            # Extract hidden states from the fine-tuned model
            output = model.language_model(**tokens, output_hidden_states=True)
            # Use the last hidden state of the first token ([CLS] or equivalent)
            embedding = output.hidden_states[-1][:, 0, :].cpu().numpy()
            embeddings.append(embedding)
    
    return np.vstack(embeddings)

def compute_shapley_values(embeddings, labels, num_samples=100, subset_size=50):
    """Approximates Shapley values using Monte Carlo sampling."""
    n = len(embeddings)
    shapley_values = np.zeros(n)
    
    for _ in tqdm(range(num_samples), desc="Computing Shapley Values"):
        subset = np.random.choice(n, subset_size, replace=False)
        subset_x, subset_y = embeddings[subset], labels[subset]
        PROXY_MODEL.fit(subset_x, subset_y)
        
        for i in subset:
            without_i = np.delete(subset, np.where(subset == i))
            x_without, y_without = embeddings[without_i], labels[without_i]
            PROXY_MODEL.fit(x_without, y_without)
            mse_without = -np.mean((PROXY_MODEL.predict(x_without) - y_without) ** 2)
            mse_with = -np.mean((PROXY_MODEL.predict(subset_x) - subset_y) ** 2)
            shapley_values[i] += (mse_with - mse_without) / num_samples
    
    return shapley_values

def select_data_based_on_shapley(df, shapley_values, keep_ratio=0.1):
    """Selects data points with the highest Shapley values."""
    num_keep = int(len(df) * keep_ratio)
    top_indices = np.argsort(shapley_values)[-num_keep:]
    return df.iloc[top_indices]

if __name__ == "__main__":
    print("Loading external dataset...")
    dataset = load_external_data()
    labels = dataset["Label"].values
    
    print("Extracting representations from target model...")
    embeddings = extract_representations(dataset)
    
    print("Computing Shapley values...")
    shapley_values = compute_shapley_values(embeddings, labels)
    
    print("Selecting important data points...")
    selected_data = select_data_based_on_shapley(dataset, shapley_values)
    
    selected_data.to_csv("datasets/TS-DShapley_selected_data.csv", index=False)
    print(f"Selected {len(selected_data)} samples. Data saved to datasets/TS-DShapley_selected_data.csv")
