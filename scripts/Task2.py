# import dependencies
import torch
import torch.nn.functional as F
import torch.utils.data.dataloader
from torch.utils.data import random_split

from transformers import AutoTokenizer, AutoModelForMaskedLM

from datasets import load_dataset

from models import MoLFormerWithRegressionHeadMLM

from utils import SMILESextra, SMILESDataset, merge_datasets

from tqdm import tqdm

import pandas as pd

# paths and global parameters 
DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
EXTERNAL_DATASET_PATH ="../tasks/External-Dataset_for_Task2.csv"
BATCH_SIZE = 1


def LiSSA(model, tokenizer, data_loader, vec, damp, repeat, depth, scale):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    vec = vec.to(device)
    ihvp = torch.zeros_like(vec, device=device)

    for r in range(repeat):
        h_est = vec.clone()

        for t, (smiles, label) in enumerate(data_loader):
            if t >= depth:
                break

            label = label.to(device).float()
            smiles_token = tokenizer(smiles, padding=True, truncation=True, return_tensors="pt")
            smiles_token = {k: v.to(device) for k, v in smiles_token.items()}

            model.zero_grad()

            outputs = model(smiles_token).squeeze()
            loss = F.mse_loss(outputs, label.squeeze())

            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)
            grads = [g if g is not None else torch.zeros_like(p, requires_grad=True) for g, p in zip(grads, model.parameters())]

            flat_grads = torch.cat([g.view(-1) for g in grads])

            hvp = torch.autograd.grad(flat_grads, model.parameters(), grad_outputs=h_est, retain_graph=True, allow_unused=True)
            hvp = [g if g is not None else torch.zeros_like(p, requires_grad=True) for g, p in zip(hvp, model.parameters())]

            hvp = torch.cat([h.view(-1) for h in hvp])

            with torch.no_grad():
                hvp = hvp + damp * h_est
                h_est = vec + h_est - hvp / scale

        ihvp += h_est / scale

    return ihvp / repeat


def compute_influence(data_loader, tokenizer, model, damp, repeat, depth, scale):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    extra_data = pd.read_csv(EXTERNAL_DATASET_PATH)
    extra_data['influence_score'] = 0.0

    for smiles, label in tqdm(data_loader, desc="Computing Influence Scores"):

        smiles = smiles[0]  
        
        label = label.to(device).float()
        smiles_token = tokenizer(smiles, padding=True, truncation=True, return_tensors="pt")
        smiles_token = {k: v.to(device) for k, v in smiles_token.items()}

        model.zero_grad()
        outputs = model(smiles_token).squeeze()
        loss = F.mse_loss(outputs, label.squeeze())

        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=False, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, model.parameters())]
        flat_grads = torch.cat([g.view(-1) for g in grads])

        ihvp = LiSSA(model, tokenizer, data_loader, flat_grads, damp, repeat, depth, scale)

        influence_score = -torch.dot(ihvp, flat_grads).item()

        # Ensure correct string comparison
        extra_data.loc[extra_data['SMILES'].astype(str) == str(smiles), 'influence_score'] = influence_score
        
    extra_data.to_csv("updated_data.csv", index=False)



def train_model(train_dataloader, test_dataloader, num_epochs):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    language_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    finetuned_language_model = torch.load("finetuned_language_model.pth", map_location="cpu")
    model = MoLFormerWithRegressionHeadMLM(finetuned_language_model)
    
    lr = 0.020104429120603076

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses = [] 
    val_losses = []  

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc= f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for smile, label in progress_bar:

            label = label.to(device).float()

            smiles_token = tokenizer(smile, padding=True, truncation=True, return_tensors="pt")
            smiles_token = {k: v.to(device) for k, v in smiles_token.items()}

            optimizer.zero_grad()
            output = model(smiles_token).squeeze()

            loss = F.mse_loss(output, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss = loss.item())

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        epoch_losses.append(avg_epoch_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for smile, label in test_dataloader:

                label = label.to(device).float()

                smiles_token = tokenizer(smile, padding=True, truncation=True, return_tensors="pt")
                smiles_token = {k: v.to(device) for k, v in smiles_token.items()}

                output = model(smiles_token).squeeze()

                loss = F.mse_loss(output, label)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_dataloader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_epoch_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

    torch.save(model, "regression_head_w_exrta_data.pth")

########################################################
# Entry point
########################################################

if __name__ == "__main__":

    compute = False

    if compute:
        # Loading tokenizer and required pretrained models from task 1
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

        language_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
        finetuned_language_model = torch.load("finetuned_language_model.pth", map_location="cpu")
        model = torch.load("regression_w_finetuning_wo_wdecay.pth", map_location="cpu")
        
        # Loading the extra dataset for task 2 and passing to dataloader
        dataset = SMILESextra(EXTERNAL_DATASET_PATH)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE , shuffle = True)

        # configurations for running the LiSSA for influence calculation
        # damp = 0.01  
        # repeat = 10   
        # depth = 50   
        # scale = 1 

        damp = 0.01  
        repeat = 1   
        depth = 1   
        scale = 1 

        influence_scores = compute_influence(dataloader, tokenizer, model, damp, repeat, depth, scale)
    
    else:
        filtered_dataset = "updated_data.csv"
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

        train_model(train_dataloader, test_dataloader, 25)

