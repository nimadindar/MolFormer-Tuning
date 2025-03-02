import torch
from torch.utils.data import ConcatDataset

import matplotlib.pyplot as plt

import pandas as pd

class SMILESDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform = None, target_transform = None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform 

    def __len__(self):
        return len(self.dataset['train'])
    
    def __getitem__(self, index):

        smiles = self.dataset['train'][index]['SMILES']
        label = self.dataset['train'][index]['label'] 

        if self.transform:
            smiles = self.transform(smiles)

        if self.target_transform:
            label = self.target_transform(label)

        return smiles, label
    
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
    

def merge_datasets(dataset1, dataset2):
    return ConcatDataset([dataset1, dataset2]) 


def loss_fig(epoch_losses, val_losses, title, save_path):

    plt.figure(figsize=(10, 6)) 
    plt.plot(epoch_losses, label="Training Loss", marker='o', linestyle='-', color='blue') 
    plt.plot(val_losses, label="Validation Loss", marker='s', linestyle='--', color='red')  

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)

    # plt.xticks(range(len(epoch_losses)))  

    plt.legend(loc="upper right", fontsize=10)

    plt.grid(alpha=0.3, linestyle='--')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()