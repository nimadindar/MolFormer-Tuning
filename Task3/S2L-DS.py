import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, DataCollatorWithPadding
from datasets import Dataset

# Define model name
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

# Check for GPU availability and use CUDA if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_external_data(file_path="datasets/External-Dataset_for_Task2.csv"):
    """Loads the external dataset from a CSV file into a Hugging Face Dataset format."""
    df = pd.read_csv(file_path)  # Read the dataset into a pandas DataFrame
    dataset = Dataset.from_pandas(df)  # Convert the DataFrame into a Dataset object
    return dataset

def compute_uncertainties(dataset, model_name=MODEL_NAME):
    """Computes prediction uncertainties for each example in the dataset."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.num_labels = 1  # Ensure model is set up for regression
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, trust_remote_code=True).to(device)
    model.eval()
    
    def tokenize_function(example):
        """Tokenizes input SMILES strings and ensures labels are retained."""
        tokens = tokenizer(example["SMILES"], padding="max_length", truncation=True, max_length=512)
        tokens["Label"] = example["Label"]  # Retain labels in the tokenized output
        return tokens
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "Label"])
    
    uncertainties = []
    
    for example in tqdm(tokenized_dataset, desc="Computing uncertainties"):
        with torch.no_grad():
            inputs = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in example.items() if k in ["input_ids", "attention_mask"]}
            outputs = model(**inputs)  # Get model outputs
            logits = outputs.logits.squeeze()
            uncertainties.append(logits[0].item() if logits.numel() > 1 else logits.item())  # Ensure single scalar value
    
    return np.array(uncertainties)

def select_high_uncertainty_data(uncertainties, dataset, selection_percentage=0.1):
    """Selects the top N% most uncertain samples."""
    num_samples_to_select = int(len(uncertainties) * selection_percentage)
    selected_indices = np.argsort(uncertainties)[-num_samples_to_select:]  # Select highest uncertainty samples
    return dataset.select(selected_indices)

if __name__ == "__main__":
    """
    Main execution block:
    - Loads external dataset
    - Computes uncertainties using model predictions
    - Selects the top uncertain data points
    - Saves the selected subset to a CSV file
    """
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    print("Loading external dataset...")
    dataset = load_external_data()
    print(f"Dataset loaded with {len(dataset)} samples.")
    
    print("Computing uncertainties...")
    uncertainties = compute_uncertainties(dataset)
    
    print("Selecting high uncertainty data points...")
    selected_data = select_high_uncertainty_data(uncertainties, dataset)
    
    df_selected = pd.DataFrame(selected_data)
    df_selected.to_csv("datasets/Uncertainty_selected_data.csv", index=False)
    print(f"Selected {len(selected_data)} samples. Data saved to datasets/Uncertainty_selected_data.csv")
