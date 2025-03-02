import torch
import torch.utils.data.dataloader
from torch.utils.data import random_split

from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset

from Influence_lissa import compute_influence

from train_model import train_model

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import SMILESextra, SMILESDataset, merge_datasets, loss_fig
from src.models import MoLFormerWithRegressionHeadMLM

from Task2_analysis import filtered_dataset

import pandas as pd


# paths and global parameters 
DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
EXTERNAL_DATASET_PATH ="../datasets/External-Dataset_for_Task2.csv"
# Batch Size is set to 1 since we want to compute the influence score for each data point
BATCH_SIZE = 1
# A parameter set to control the flow of code - if set to True the code will compute the 
# influence scores else it will train the model using the new dataset. 
compute = False 

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load the finetuned language model (Finetuned using Masked Language Modeling)
language_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
finetuned_language_model = torch.load("../weights/finetuned_language_model.pth", map_location="cpu")

model = MoLFormerWithRegressionHeadMLM(finetuned_language_model)



if __name__ == "__main__":

    if compute:

        model = torch.load("../weights/regression_w_finetuning_wo_wdecay.pth", map_location="cpu")
        
        # Loading the extra dataset for task 2 and passing to dataloader
        dataset = SMILESextra(EXTERNAL_DATASET_PATH)
        extra_data = pd.read_csv(EXTERNAL_DATASET_PATH)

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

        # The pandas data frame is being passed as input since we want to alter this data frame and add a column.
        influence_scores = compute_influence(extra_data, dataloader, tokenizer, model, damp, repeat, depth, scale)
    
    else:

        filtered_dataset(pd.read_csv("../datasets/updated_data_w_influence_scores.csv"))
        filtered_dataset_path = "../datasets/filtered_extrapoints.csv"

        dataset = load_dataset(DATASET_PATH)

        dataset_main = SMILESDataset(dataset)
        dataset_extra = SMILESextra(filtered_dataset_path)

        merged_dataset = merge_datasets(dataset_main, dataset_extra)

        train_size = int(0.8 * len(merged_dataset))
        test_size = len(merged_dataset) - train_size

        train_smiles, test_smiles = random_split(merged_dataset, [train_size, test_size])

        train_dataloader = torch.utils.data.DataLoader(train_smiles, batch_size=16, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_smiles, batch_size=16, shuffle = True)

        epoch_losses, val_losses = train_model(model, tokenizer, train_dataloader, test_dataloader, num_epochs = 1)

        loss_fig(epoch_losses, val_losses,
                title = "Train Loss vs. Validation Loss for Model trained on Extra Dataset", 
                save_path = "task2.png")

