import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from src.models import MoLFormerWithRegressionHeadMLM
from src.utils import SMILESDataset, SMILESextra, merge_datasets, loss_fig

# Define LoRA layer
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Low-rank matrices A and B for LoRA adaptation
        self.A = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.scaling = self.alpha / self.rank

    def forward(self, x):
        # Apply LoRA transformation: x + scaling * (x @ B^T @ A^T)
        return x + (self.scaling * (x @ self.B.T @ self.A.T))

# Define LoRA-enabled MoLFormer model
class LoRAMoLFormer(nn.Module):
    def __init__(self, model, rank=8, alpha=16):
        super().__init__()
        self.model = model
        
        # Freeze all pretrained model parameters
        for param in self.model.language_model.parameters():
            param.requires_grad = False

        # Apply LoRA to attention layers (query and value projection layers)
        for name, module in self.model.language_model.named_modules():
            if "query" in name or "value" in name:
                parent_module = self.get_parent_module(name)
                setattr(parent_module, name.split('.')[-1], LoRALayer(module.in_features, module.out_features, rank, alpha))

    def forward(self, token):
        return self.model(token)

    def get_parent_module(self, module_name):
        """Helper function to retrieve the parent module of a given submodule."""
        components = module_name.split(".")
        parent = self.model.language_model
        for comp in components[:-1]:  # Traverse to the parent module
            parent = getattr(parent, comp)
        return parent

# Training function
def train_model(train_dataloader, test_dataloader, num_epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    raw_language_model = AutoModel.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)
    
    # Initialize MoLFormer with LoRA
    model = MoLFormerWithRegressionHeadMLM(raw_language_model)
    lora_model = LoRAMoLFormer(model).to(device)
    
    # Define optimizer and learning rate
    lr = 0.020104429120603076
    optimizer = torch.optim.Adam(lora_model.parameters(), lr=lr)
    
    epoch_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        lora_model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for smile, label in progress_bar:
            label = label.to(device).float()
            smiles_token = tokenizer(smile, padding=True, truncation=True, return_tensors="pt")
            smiles_token = {k: v.to(device) for k, v in smiles_token.items()}
            
            optimizer.zero_grad()
            output = lora_model(smiles_token).squeeze()
            
            loss = F.mse_loss(output, label)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        epoch_losses.append(avg_epoch_loss)
        
        # Validation loop
        lora_model.eval()
        val_loss = 0
        with torch.no_grad():
            for smile, label in test_dataloader:
                label = label.to(device).float()
                smiles_token = tokenizer(smile, padding=True, truncation=True, return_tensors="pt")
                smiles_token = {k: v.to(device) for k, v in smiles_token.items()}
                output = lora_model(smiles_token).squeeze()
                loss = F.mse_loss(output, label)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_dataloader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_epoch_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")
    
    return epoch_losses, val_losses

if __name__ == "__main__":
    # Load datasets
    filtered_dataset = "F:/Saarland University Courses/Neural Networks/all files/Project/Project-GitHub-NimaB/NNTI_project/datasets/filtered_extrapoints.csv"
    DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
    dataset = load_dataset(DATASET_PATH)
    
    dataset_main = SMILESDataset(dataset)
    dataset_extra = SMILESextra(filtered_dataset)
    
    # Merge main and extra datasets
    merged_dataset = merge_datasets(dataset_main, dataset_extra)
    train_size = int(0.8 * len(merged_dataset))
    test_size = len(merged_dataset) - train_size
    
    train_smiles, test_smiles = random_split(merged_dataset, [train_size, test_size])
    
    # Prepare DataLoaders
    train_dataloader = DataLoader(train_smiles, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_smiles, batch_size=16, shuffle=False)
    
    # Train the LoRA model
    epoch_losses, test_losses = train_model(train_dataloader, test_dataloader, num_epochs=40)
    
    # Save the training and validation loss plot
    loss_fig(epoch_losses, test_losses, 
             "Training & Validation Loss Over Epochs for LoRA Method",
             save_path='LoRA_val_vs_train_loss.png')
