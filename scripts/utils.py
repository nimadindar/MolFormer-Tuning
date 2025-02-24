import torch

import pandas as pd

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