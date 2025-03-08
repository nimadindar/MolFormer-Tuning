import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import SMILESextra, SMILESDataset, merge_datasets, loss_fig
from src.models import MoLFormerWithRegressionHeadMLM

class BitFitMoLFormer(nn.Module):
    """
    Wrapper model for BitFit, which freezes all parameters except for biases in the pre-trained model.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # Freeze all parameters except biases
        for name, param in self.model.language_model.named_parameters():
            if "bias" not in name:
                param.requires_grad = False

    def forward(self, token):
        return self.model(token)
    
class BitFitMolFormerMI(nn.Module):
    """
    Wrapper model for BitFit, which freezes all parameters except for biases in the pre-trained model.
    This class is implemented for multi-input model which accepts tokens and feature vector as input for 
    forward method.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # Freeze all parameters except biases
        for name, param in self.model.language_model.named_parameters():
            if "bias" not in name:
                param.requires_grad = False

    def forward(self, token, feature_vector):
        return self.model(token, feature_vector)



def train_model(train_dataloader, test_dataloader, num_epochs):
    """
    Trains the BitFitMoLFormer model using Mean Squared Error (MSE) loss.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer and base model
    MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    raw_language_model = AutoModel.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)
    
    # Initialize model with regression head
    model = MoLFormerWithRegressionHeadMLM(raw_language_model)
    bitfit_model = BitFitMoLFormer(model).to(device)
    
    # Define optimizer with pre-selected learning rate
    lr = 0.001 
    optimizer = torch.optim.Adam(bitfit_model.parameters(), lr=lr)
    
    epoch_losses = []  # Track training losses
    val_losses = []  # Track validation losses

    for epoch in range(num_epochs):
        bitfit_model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for smile, label in progress_bar:
            label = label.to(device).float()
            
            # Tokenize SMILES strings
            smiles_token = tokenizer(smile, padding=True, truncation=True, return_tensors="pt")
            smiles_token = {k: v.to(device) for k, v in smiles_token.items()}
            
            # Forward pass
            optimizer.zero_grad()
            output = bitfit_model(smiles_token).squeeze()
            
            # Compute loss and backpropagate
            loss = F.mse_loss(output, label)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Compute average training loss
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        epoch_losses.append(avg_epoch_loss)
        
        # Validation phase
        bitfit_model.eval()
        val_loss = 0
        with torch.no_grad():
            for smile, label in test_dataloader:
                label = label.to(device).float()
                smiles_token = tokenizer(smile, padding=True, truncation=True, return_tensors="pt")
                smiles_token = {k: v.to(device) for k, v in smiles_token.items()}
                output = bitfit_model(smiles_token).squeeze()
                loss = F.mse_loss(output, label)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_dataloader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_epoch_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")
    
    return epoch_losses, val_losses


if __name__ == "__main__":
    """
    Main execution block:
    - Loads datasets (main and external)
    - Merges datasets and splits into training and testing sets
    - Trains the BitFit model
    - Saves the training and validation loss plot
    """
    filtered_dataset_path = "../datasets/Uncertainty_selected_data.csv"
    DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
    
    # Load datasets
    dataset = load_dataset(DATASET_PATH)
    dataset_main = SMILESDataset(dataset)
    dataset_extra = SMILESextra(filtered_dataset_path)
    
    # Merge main and extra datasets
    merged_dataset = merge_datasets(dataset_main, dataset_extra)
    
    # Split into training and test sets (80% training, 20% testing)
    train_size = int(0.8 * len(merged_dataset))
    test_size = len(merged_dataset) - train_size
    train_smiles, test_smiles = random_split(merged_dataset, [train_size, test_size])
    
    # Create data loaders
    train_dataloader = DataLoader(train_smiles, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_smiles, batch_size=16, shuffle=True)
    
    # Train the model and visualize loss trends
    epoch_losses, test_losses = train_model(train_dataloader, test_dataloader, num_epochs=40)
    loss_fig(epoch_losses, test_losses, 
             "Training & Validation Loss Over Epochs for BitFit Method",
             save_path='BitFit_val_vs_train_loss.png')
