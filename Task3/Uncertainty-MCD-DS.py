import torch
import numpy as np
import pandas as pd
import os
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoConfig
)
from datasets import load_dataset, Dataset
from sklearn.metrics import mean_squared_error
from src.models import MoLFormerWithRegressionHeadMLM  # Custom model import

# Set environment variables
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load dataset
dataset = load_dataset(DATASET_PATH)

# Ensure correct column name for SMILES representation
smiles_column = "smiles" if "smiles" in dataset["train"].column_names else "SMILES"

def tokenize_function(examples):
    return tokenizer(examples[smiles_column], padding="max_length", truncation=True, max_length=512)

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load model configuration and modify for regression
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
config.num_labels = 1  # Regression task

# Load custom model
model = MoLFormerWithRegressionHeadMLM.from_pretrained(MODEL_NAME, config=config).to(device)
model.train()  # Keep dropout enabled for uncertainty estimation

def compute_uncertainties(dataset, model, num_samples=10):
    """Computes epistemic uncertainty using Monte Carlo Dropout."""
    uncertainties = []
    with torch.no_grad():
        for example in dataset:
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(device)

            predictions = []
            for _ in range(num_samples):
                outputs = model(input_ids, attention_mask=attention_mask)
                predictions.append(outputs.logits.squeeze().cpu().numpy())

            uncertainties.append(np.var(predictions, axis=0))  # Variance as uncertainty measure

    return np.array(uncertainties)

# Compute uncertainties
uncertainties = compute_uncertainties(tokenized_datasets["train"], model)

# Select top 10% most uncertain samples
selection_percentage = 0.1
num_samples_to_select = int(len(uncertainties) * selection_percentage)
selected_indices = np.argsort(uncertainties)[-num_samples_to_select:]
selected_dataset = tokenized_datasets["train"].select(selected_indices)

# Convert to pandas DataFrame and save
selected_df = pd.DataFrame(selected_dataset)
selected_df.to_csv("datasets/Uncertainty_MCD_selected_data.csv", index=False)
print(f"Selected {len(selected_dataset)} samples. Data saved to datasets/Uncertainty_MCD_selected_data.csv")

# Ensure no duplicates in merged dataset
selected_smiles = set(selected_dataset[smiles_column])
original_smiles = set(dataset["train"][smiles_column])

new_samples = [i for i in selected_dataset if i[smiles_column] not in original_smiles]

# Convert to Dataset
selected_dataset = Dataset.from_dict({
    smiles_column: [sample[smiles_column] for sample in new_samples],
    "Label": [sample["Label"] for sample in new_samples]
})

# Merge datasets without duplicates
filtered_dataset = dataset["train"].filter(lambda example: example[smiles_column] not in selected_smiles)
full_dataset = filtered_dataset.add(selected_dataset)

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=1
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=full_dataset,
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=lambda eval_pred: {"mse": mean_squared_error(eval_pred.label_ids, eval_pred.predictions.squeeze())},
)

# Train model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)
