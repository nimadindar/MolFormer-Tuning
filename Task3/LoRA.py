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

# Define LoRA layer
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.scaling = self.alpha / self.rank

    def forward(self, x):
        return x + (self.scaling * (x @ self.B.T @ self.A.T))

class MoLFormerWithLoRA(nn.Module):
    def __init__(self, language_model, rank=8, alpha=16):
        super().__init__()
        self.language_model = language_model

        # Collect target modules first
        modules_to_replace = {}
        for name, module in self.language_model.named_modules():
            if "query" in name or "value" in name:
                modules_to_replace[name] = LoRALayer(module.in_features, module.out_features, rank, alpha)

        # Apply replacements after iteration
        for name, new_module in modules_to_replace.items():
            parent_module = self.get_parent_module(self.language_model, name)
            setattr(parent_module, name.split('.')[-1], new_module)

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

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask)  
        embeddings = outputs.last_hidden_state[:, 0, :].detach().float()
        logits = self.regression_head(embeddings)

        if labels is not None:
            loss = F.mse_loss(logits.squeeze(), labels.float())
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

    def get_parent_module(self, model, module_name):
        """Helper function to retrieve the parent module of a given submodule."""
        components = module_name.split(".")
        parent = model
        for comp in components[:-1]:  # Skip the last part (actual module name)
            parent = getattr(parent, comp)
        return parent


def train_lora():
    dataset = load_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    language_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = MoLFormerWithLoRA(language_model).to(device)

    def tokenize_function(examples):
        tokenized = tokenizer(examples["SMILES"], padding="max_length", truncation=True, max_length=512)
        
        if "label" not in examples or examples["label"] is None:
            return None  # This ensures missing examples are skipped
        
        tokenized["labels"] = [float(label) if label is not None else 0.0 for label in examples["label"]]
        return tokenized

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_datasets = tokenized_datasets.filter(lambda x: x is not None)  # Remove invalid samples
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir="./lora_model",
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

    wandb.init(project="MoleculeNet-FineTuning-LoRA")  # Move W&B before training
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
    print("LoRA fine-tuning completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    train_lora()
