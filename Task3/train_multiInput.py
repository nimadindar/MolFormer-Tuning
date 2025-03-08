import torch
import torch.nn.functional as F
from torch.utils.data import random_split

from mol_prop import calculate_descriptions

from datasets import load_dataset

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import SMILESextra, SMILESDataset, merge_datasets, loss_fig
from src.models import MultiInputModel
from transformers import AutoModel, AutoTokenizer

from BitFit import BitFitMolFormerMI

from tqdm import tqdm



def train_model(train_dataloader, test_dataloader, num_epochs):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    raw_language_model = AutoModel.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)

    model = MultiInputModel(raw_language_model)

    bitfit_model = BitFitMolFormerMI(model)
    
    lr = 0.001

    optimizer = torch.optim.Adam(bitfit_model.parameters(), lr=lr)

    epoch_losses = [] 
    val_losses = []  

    for epoch in range(num_epochs):
        bitfit_model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc= f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for smile, label in progress_bar:

            label = label.to(device).float()

            feature_vectors = calculate_descriptions(smile)
            feature_vectors.to(device)

            smiles_token = tokenizer(smile, padding=True, truncation=True, return_tensors="pt")
            smiles_token = {k: v.to(device) for k, v in smiles_token.items()}

            optimizer.zero_grad()
            output = bitfit_model(smiles_token, feature_vectors).squeeze()

            loss = F.mse_loss(output, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss = loss.item())

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        epoch_losses.append(avg_epoch_loss)

        bitfit_model.eval()
        val_loss = 0
        with torch.no_grad():
            for smile, label in test_dataloader:

                label = label.to(device).float()

                feature_vectors = calculate_descriptions(smile)
                feature_vectors.to(device)

                smiles_token = tokenizer(smile, padding=True, truncation=True, return_tensors="pt")
                smiles_token = {k: v.to(device) for k, v in smiles_token.items()}

                output = bitfit_model(smiles_token, feature_vectors).squeeze()

                loss = F.mse_loss(output, label)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_dataloader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_epoch_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

    # torch.save(ia3_model, "IA3_model_test.pth")

    return epoch_losses, val_losses



filtered_dataset = "../datasets/Uncertainty_selected_data.csv"
DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"

dataset = load_dataset(DATASET_PATH)

dataset_main = SMILESDataset(dataset)
dataset_extra = SMILESextra(filtered_dataset)

merged_dataset = merge_datasets(dataset_main, dataset_extra)

train_size = int(0.8 * len(merged_dataset))
test_size = len(merged_dataset) - train_size

train_smiles, test_smiles = random_split(merged_dataset, [train_size, test_size])

train_dataloader = torch.utils.data.DataLoader(train_smiles, batch_size=16, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_smiles, batch_size=16, shuffle = True)

epoch_losses, test_losses = train_model(train_dataloader, test_dataloader, 40)
loss_fig(epoch_losses, test_losses, 
            "Training & Validation Loss Over Epochs for BitFit method with Multi Input Model",
            save_path='BitFit_multiInput_val_vs_train_loss.png')
        