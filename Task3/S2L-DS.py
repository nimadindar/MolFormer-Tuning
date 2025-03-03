import wandb
import torch
import datasets
import argparse
import numpy as np
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.cluster import KMeans
from datasets import load_dataset
import pandas as pd


MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"  # Large model
PROXY_MODEL_NAME = "EleutherAI/pythia-70m"  # Small model for loss tracking
DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"  # Main dataset

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_data():
    """Loads the MoleculeNet Lipophilicity dataset."""
    dataset = load_dataset(DATASET_PATH, download_mode="force_redownload")
    return dataset["train"]

def compute_loss_trajectories(dataset, model_name=PROXY_MODEL_NAME, num_samples=5000):
    """Trains a small proxy model and records loss trajectories."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
    model.resize_token_embeddings(len(tokenizer))  # Ensure the model recognizes the new padding token

    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    def tokenize_function(example):
        return tokenizer(example["SMILES"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # Ensures proper padding
    
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    training_args = TrainingArguments(
        output_dir="./proxy_model",
        per_device_train_batch_size=1,  # Lowered for 4GB GPU
        num_train_epochs=1,
        save_strategy="no",
        logging_steps=100,
        fp16=True,  # Enable mixed precision
        gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
    )
    
    # Ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"

    # Resize model embeddings to include the new pad token
    model.resize_token_embeddings(len(tokenizer))



    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,  # Use dynamic padding
        tokenizer=tokenizer
    )
    trainer.train()
    
    loss_trajectories = []
    for example in tokenized_dataset:
        with torch.no_grad():
            inputs = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in example.items() if k in ["input_ids", "attention_mask"]}
            outputs = model(**inputs)
            loss = torch.nn.functional.mse_loss(outputs.logits.squeeze(), torch.tensor(example["label"]).float().to(device))
            loss_trajectories.append(loss.item())
    
    return np.array(loss_trajectories)

def cluster_and_select_data(loss_trajectories, dataset, num_clusters=10, num_samples=1000):
    """Clusters examples based on loss trajectories and selects a balanced subset."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(loss_trajectories.reshape(-1, 1))
    
    selected_indices = []
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        num_to_select = min(len(cluster_indices), num_samples // num_clusters)
        selected_indices.extend(np.random.choice(cluster_indices, num_to_select, replace=False))
    
    return dataset.select(selected_indices)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    wandb.init(project="MoleculeNet-FineTuning")
    
    dataset = load_data()
    loss_trajectories = compute_loss_trajectories(dataset)
    selected_data = cluster_and_select_data(loss_trajectories, dataset)

    # Print the first 5 selected data points
    print("Selected Data Points (First 5):")
    for idx, example in enumerate(selected_data[:5]):  # Printing first 5 selected samples
        print(f"Sample {idx}: SMILES: {example['SMILES']}, Label: {example['label']}")

    # Optionally, save the selected data to a CSV file
    df = pd.DataFrame(selected_data)
    df.to_csv("selected_data.csv", index=False)
    print("Data saved to selected_data.csv")
    
    wandb.log({"selected_data_size": len(selected_data)})
    
    print(f"Selected {len(selected_data)} samples using S2L data selection strategy.")
    
