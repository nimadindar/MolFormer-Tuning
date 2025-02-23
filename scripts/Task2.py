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
    

def LiSSA(model, dataloader, tokenizer, T, S1, S2, T1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 0.020104429120603076

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for param in model.parameters():
        param.requires_grad = True

    # First Order Optimization warm-up
    for _ in range(T1):
        for smiles, label in dataloader:
            label = label.to(device).float()

            # Tokenize input and move to device
            smiles_token = tokenizer(smiles, padding=True, truncation=True, return_tensors="pt")
            smiles_token = {k: v.to(device) for k, v in smiles_token.items()}

            optimizer.zero_grad()
            output = model(smiles_token)
            loss = F.mse_loss(output.squeeze(), label)
            loss.backward()
            optimizer.step()
    
    influence_scores = {}

    # Main LiSSA loop
    for t in range(T):
        for smiles, label in dataloader:
            label = label.to(device).float()

            # Tokenize input and move to device
            smiles_token = tokenizer(smiles, padding=True, truncation=True, return_tensors="pt")
            smiles_token = {k: v.to(device) for k, v in smiles_token.items()}

            optimizer.zero_grad()

            # Forward pass
            output = model(smiles_token)
            loss = F.mse_loss(output.squeeze(), label)

            # Compute first-order gradients
            grad_f = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)
            grad_f = [g if g is not None else torch.zeros_like(p, requires_grad=True) for g, p in zip(grad_f, model.parameters())]

            X = []
            for i in range(S1):
                X_i = [g.clone().detach() for g in grad_f]

                for j in range(S2):
                    # Compute Hessian-vector product
                    hvp_sample = torch.autograd.grad(
                        outputs=grad_f,
                        inputs=model.parameters(),
                        grad_outputs=[torch.ones_like(g) for g in grad_f],
                        retain_graph=True,
                        allow_unused = True
                    )

   
                    X_i = [g1 + (torch.eye(g2.shape[0]).to(device) - g2) * g3 
                        for g1, g2, g3 in zip(grad_f, hvp_sample, X_i)]

            
            Xt = [sum(X_i[k] for X_i in X) / S1 for k in range(len(X[0]))]
            influence_scores[(smiles, label)] = Xt

    return influence_scores


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


    influences = LiSSA(model, dataloader, tokenizer, 2, 2, 2, 2)

    print(influences)



