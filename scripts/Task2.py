# import dependencies
import torch
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling

import torch.nn.functional as F

# import sklearn
# import datasets
# import numpy as np
import torch.utils.data.dataloader
# import transformers
import pandas as pd
# from tqdm import tqdm
from transformers import AutoTokenizer



DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"


class MoLFormerWithRegressionHeadMLM(nn.Module):
    def __init__(self, language_model):
        super().__init__()

        self.language_model = language_model
        
        for param in self.language_model.parameters():
            param.requires_grad = True

        self.layer = nn.Sequential(
            # Layer 1
            nn.Linear(768, 407),
            nn.BatchNorm1d(407),
            nn.ELU(),
            nn.Dropout(0.3819889369374411),
            # Layer 2
            nn.Linear(407,427),
            nn.BatchNorm1d(427),
            nn.ELU(),
            nn.Dropout(0.2777819584361112),
            # Layer 3
            nn.Linear(427,240),
            nn.BatchNorm1d(240),
            nn.ELU(),
            nn.Dropout(0.4619253799146514),
            # Layer 4
            nn.Linear(240, 69),
            nn.BatchNorm1d(69),
            nn.ELU(),
            nn.Dropout(0.4497051723910867),
            nn.Linear(69,1)
        )
    
    def forward(self, token):

        smiles_output = self.language_model(**token, output_hidden_states=True)
        smiles_embedding = smiles_output.hidden_states[-1][:, 0, :].float()
        
        return self.layer(smiles_embedding)

class SMILESextra(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = pd.read_csv(self.dataset_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        smiles = self.data.iloc[index]['SMILES']
        label = self.data.iloc[index]['Label']

        return smiles, label
    



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

            outputs = model(smiles_token)
            loss = F.mse_loss(outputs, label)
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

########################################################
# Entry point
########################################################

if __name__ == "__main__":

    dataset_path="../tasks/External-Dataset_for_Task2.csv"

    language_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    finetuned_language_model = torch.load("finetuned_language_model.pth", map_location="cpu")

    model = torch.load("regression_w_finetuning_wo_wdecay.pth", map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


    dataset = SMILESextra(dataset_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 16, shuffle = True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Generate vec from a test batch
    smiles_batch, labels_batch = next(iter(dataloader))
    labels_batch = labels_batch.to(device).float()
    smiles_tokens = tokenizer(smiles_batch, padding=True, truncation=True, return_tensors="pt")
    smiles_tokens = {k: v.to(device) for k, v in smiles_tokens.items()}

    # Compute gradients to form vec
    model.zero_grad()
    outputs = model(smiles_tokens)
    loss = F.mse_loss(outputs, labels_batch)
    grads = torch.autograd.grad(loss, model.parameters(), allow_unused=True)
    grads = [g if g is not None else torch.zeros_like(p, requires_grad=True) for g, p in zip(grads, model.parameters())]

    vec = torch.cat([g.detach().view(-1) for g in grads])

    # Run LiSSA algorithm
    damp = 0.01  # Damping factor
    repeat = 5   # Number of repeats
    depth = 10   # Recurrence depth
    scale = 10.0 # Scaling factor

    result = LiSSA(model, tokenizer, dataloader, vec, damp, repeat, depth, scale)

    # Print the result
    print("Inverse Hessian-Vector Product (IHVP) result:", result)



