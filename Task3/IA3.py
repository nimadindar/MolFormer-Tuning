import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

from src.models import MoLFormerWithRegressionHeadMLM
from transformers import AutoModel, AutoTokenizer

from datasets import load_dataset
from src.utils import SMILESDataset, SMILESextra, merge_datasets, loss_fig

from tqdm import tqdm

class IA3MolFormer(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model

        for param in self.model.language_model.parameters():
            param.requires_grad = False
        
        self.modify_attention_layers()

    def modify_attention_layers(self):

        for layer in self.model.language_model.encoder.layer:
            attn = layer.attention.self


            attn.ia3_alpha_k = nn.Parameter(torch.ones(attn.key.out_features, 1)) 
            attn.ia3_alpha_v = nn.Parameter(torch.ones(attn.value.out_features, 1))  

            def modify_kv_projection(original_forward, alpha):
                return lambda x: original_forward(x) * alpha.T 
            
            attn.key.forward = modify_kv_projection(attn.key.forward, attn.ia3_alpha_k)
            attn.value.forward = modify_kv_projection(attn.value.forward, attn.ia3_alpha_v)

    def forward(self, token):
        return self.model(token)

class IA3MolFormerMI(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model

        for param in self.model.language_model.parameters():
            param.requires_grad = False
        
        self.modify_attention_layers()

    def modify_attention_layers(self):

        for layer in self.model.language_model.encoder.layer:
            attn = layer.attention.self


            attn.ia3_alpha_k = nn.Parameter(torch.ones(attn.key.out_features, 1)) 
            attn.ia3_alpha_v = nn.Parameter(torch.ones(attn.value.out_features, 1))  

            def modify_kv_projection(original_forward, alpha):
                return lambda x: original_forward(x) * alpha.T 
            
            attn.key.forward = modify_kv_projection(attn.key.forward, attn.ia3_alpha_k)
            attn.value.forward = modify_kv_projection(attn.value.forward, attn.ia3_alpha_v)

    def forward(self, token, feature_vector):
        return self.model(token, feature_vector)

def train_model(train_dataloader, test_dataloader, num_epochs):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    raw_language_model = AutoModel.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)

    model = MoLFormerWithRegressionHeadMLM(raw_language_model)

    ia3_model = IA3MolFormer(model)
    
    lr = 0.020104429120603076

    optimizer = torch.optim.Adam(ia3_model.parameters(), lr=lr)

    epoch_losses = [] 
    val_losses = []  

    for epoch in range(num_epochs):
        ia3_model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc= f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for smile, label in progress_bar:

            label = label.to(device).float()

            smiles_token = tokenizer(smile, padding=True, truncation=True, return_tensors="pt")
            smiles_token = {k: v.to(device) for k, v in smiles_token.items()}

            optimizer.zero_grad()
            output = ia3_model(smiles_token).squeeze()

            loss = F.mse_loss(output, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss = loss.item())

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        epoch_losses.append(avg_epoch_loss)

        ia3_model.eval()
        val_loss = 0
        with torch.no_grad():
            for smile, label in test_dataloader:

                label = label.to(device).float()

                smiles_token = tokenizer(smile, padding=True, truncation=True, return_tensors="pt")
                smiles_token = {k: v.to(device) for k, v in smiles_token.items()}

                output = ia3_model(smiles_token).squeeze()

                loss = F.mse_loss(output, label)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_dataloader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_epoch_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

    # torch.save(ia3_model, "IA3_model_test.pth")

    return epoch_losses, val_losses




if __name__ == "__main__":

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

        epoch_losses, test_losses = train_model(train_dataloader, test_dataloader, 1)
        loss_fig(epoch_losses, test_losses, 
                 "Training & Validation Loss Over Epochs for IA3 method",
                 save_path='IA3_val_vs_train_loss.png')
        

