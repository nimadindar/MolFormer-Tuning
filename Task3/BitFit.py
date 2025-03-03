import torch
import wandb
import argparse
import transformers
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_data():
    dataset = load_dataset(DATASET_PATH, download_mode="force_redownload")
    return dataset["train"].train_test_split(test_size=0.1)

def freeze_all_except_bias(model):
    for name, param in model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

class MoLFormerWithRegressionHead(nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model
        self.regression_head = nn.Sequential(
            nn.Linear(768, 407),
            nn.BatchNorm1d(407),
            nn.ELU(),
            nn.Dropout(0.38),
            nn.Linear(407, 427),
            nn.BatchNorm1d(427),
            nn.ELU(),
            nn.Dropout(0.27),
            nn.Linear(427, 240),
            nn.BatchNorm1d(240),
            nn.ELU(),
            nn.Dropout(0.46),
            nn.Linear(240, 69),
            nn.BatchNorm1d(69),
            nn.ELU(),
            nn.Dropout(0.44),
            nn.Linear(69, 1)
        )
    
    def forward(self, input_ids, attention_mask, labels=None):  # Accept input_ids & attention_mask
        outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask)  
        embeddings = outputs.last_hidden_state[:, 0, :].detach().float()
        logits = self.regression_head(embeddings)

        if labels is not None:  # Compute loss inside forward pass
            loss = F.mse_loss(logits.squeeze(), labels.float())
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


def train_bitfit():
    dataset = load_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    language_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = MoLFormerWithRegressionHead(language_model).to(device)
    freeze_all_except_bias(model)

    def tokenize_function(examples):
        tokenized = tokenizer(examples["SMILES"], padding="max_length", truncation=True, max_length=512)

        # Ensure no missing labels (replace None with 0.0 or filter them out)
        if "label" not in examples or examples["label"] is None:
            return None  # This ensures missing examples are skipped

        tokenized["labels"] = [float(label) if label is not None else 0.0 for label in examples["label"]]
        return tokenized



    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_datasets = tokenized_datasets.filter(lambda x: x is not None)  # Remove invalid samples

    # Correct set_format
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])



    training_args = TrainingArguments(
        output_dir="./bitfit_model",
        per_device_train_batch_size=4,  # Reduce batch size for better multi-GPU performance
        num_train_epochs=3,
        save_strategy="epoch",
        logging_steps=100,  # Reduce logging frequency
        fp16=True,
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    wandb.init(project="MoleculeNet-FineTuning-BitFit")  # Move W&B before training
    print(tokenized_datasets["train"][:5])  # Check first 5 samples

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )
    
    trainer.train()

    wandb.log({"final_eval_loss": trainer.evaluate()["eval_loss"]})
    print("BitFit fine-tuning completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    train_bitfit()
