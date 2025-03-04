import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from sklearn.cluster import KMeans
from datasets import Dataset

# Define model names
PROXY_MODEL_NAME = "EleutherAI/pythia-70m"  # Small model used to track loss trajectories

# Check for GPU availability and use CUDA if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_external_data(file_path="datasets/External-Dataset_for_Task2.csv"):
    """Loads the external dataset from a CSV file into a Hugging Face Dataset format."""

    df = pd.read_csv(file_path)  # Read the dataset into a pandas DataFrame
    dataset = Dataset.from_pandas(df)  # Convert the DataFrame into a Dataset object
    return dataset

def compute_loss_trajectories(dataset, model_name=PROXY_MODEL_NAME):
    """Trains a small proxy model and records loss trajectories for each example in the dataset."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load the tokenizer for the given model
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"  # Set padding token
    
    # Load the model for regression-based sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
    model.config.problem_type = "regression"  # Ensure model uses regression loss
    model.resize_token_embeddings(len(tokenizer))  # Resize token embeddings to fit the tokenizer
    
    def tokenize_function(example):
        """Tokenizes input SMILES strings and ensures labels are retained."""

        tokens = tokenizer(example["SMILES"], padding="max_length", truncation=True, max_length=512)
        tokens["Label"] = example["Label"]  # Retain labels in the tokenized output
        return tokens
    
    # Apply tokenization to the entire dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # Data collator for padding
    
    # Convert dataset into torch format with required columns
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "Label"])
    
    loss_trajectories = []  # List to store loss values for each example
    
    # Compute loss for each example in the dataset
    for example in tqdm(tokenized_dataset, desc="Computing loss trajectories"):
        with torch.no_grad():  # Disable gradient computation for inference
            inputs = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in example.items() if k in ["input_ids", "attention_mask"]}  # Prepare input tensors
            label = torch.tensor(example["Label"]).float().to(device).unsqueeze(0)  # Convert label to tensor
            outputs = model(**inputs)  # Get model outputs
            loss = torch.nn.functional.mse_loss(outputs.logits.squeeze(), label)  # Compute Mean Squared Error (MSE) loss
            loss_trajectories.append(loss.item())  # Store computed loss value
    
    return np.array(loss_trajectories)  # Return loss values as a NumPy array

def cluster_and_select_data(loss_trajectories, dataset, num_clusters=10, num_samples=100):
    """Clusters examples based on loss trajectories and selects a balanced subset from each cluster."""

    # Normalize loss values to have zero mean and unit variance
    loss_trajectories = (loss_trajectories - np.mean(loss_trajectories)) / np.std(loss_trajectories)
    
    # Perform KMeans clustering on the loss values
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(loss_trajectories.reshape(-1, 1))  # Assign each sample to a cluster
    
    selected_indices = []  # List to store selected sample indices
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]  # Get all samples in the current cluster
        num_to_select = min(len(cluster_indices), num_samples // num_clusters)  # Determine number of samples to select per cluster
        selected_indices.extend(np.random.choice(cluster_indices, num_to_select, replace=False))  # Randomly select samples
    
    return dataset.select(selected_indices)  # Return selected subset of dataset

if __name__ == "__main__":
    """
    Main execution block:
    - Loads external dataset
    - Computes loss trajectories using a small proxy model
    - Clusters samples based on loss and selects a balanced subset
    - Saves the selected subset to a CSV file
    """
    # Parse command-line arguments (currently not used but allows for future extensions)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    print("Loading external dataset...")
    dataset = load_external_data()  # Load dataset from CSV file
    print(f"Dataset loaded with {len(dataset)} samples.")
    
    print("Computing loss trajectories...")
    loss_trajectories = compute_loss_trajectories(dataset)  # Compute loss values for dataset
    
    print("Selecting important data points using S2L...")
    selected_data = cluster_and_select_data(loss_trajectories, dataset)  # Select diverse and informative samples
    
    # Convert selected dataset to a pandas DataFrame and save as CSV
    df_selected = pd.DataFrame(selected_data)
    df_selected.to_csv(r"F:\Saarland University Courses\Neural Networks\all files\Project\Project-GitHub-NimaB\NNTI_project\datasets\S2L_selected_data.csv", index=False)
    print(f"Selected {len(selected_data)} samples. Data saved to datasets/S2L_selected_data.csv")
