import torch
from torch.utils.data import ConcatDataset

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