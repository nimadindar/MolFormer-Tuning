import torch
import numpy as np
import os
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding, 
    AutoConfig
)
from datasets import load_dataset
from sklearn.metrics import mean_squared_error

# Set environment variables
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Constants
DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load dataset
dataset = load_dataset(DATASET_PATH)

# Check dataset column names
print("Dataset columns:", dataset["train"].column_names)

# Ensure correct column name for SMILES representation
smiles_column = "smiles" if "smiles" in dataset["train"].column_names else "SMILES"

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples[smiles_column], padding="max_length", truncation=True, max_length=512)

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load model configuration and modify for regression
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
config.num_labels = 1  # Set model to regression mode

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config, trust_remote_code=True)

# Compute uncertainty using variance
def compute_uncertainty(predictions):
    return np.var(predictions, axis=0)

# Compute metrics for regression
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    mse = mean_squared_error(labels, predictions)
    return {"mse": mse}

if __name__ == "__main__":
    model.eval()
    uncertainties = []

    with torch.no_grad():
        for example in tokenized_datasets["train"]:
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0)
            attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze().item()
            uncertainties.append(logits)

    uncertainties = compute_uncertainty(uncertainties)

    # Select top 10% most uncertain samples
    selection_percentage = 0.1
    num_samples_to_select = int(len(uncertainties) * selection_percentage)
    selected_indices = np.argsort(uncertainties)[-num_samples_to_select:]
    selected_dataset = tokenized_datasets["train"].select(selected_indices)

    # Training arguments
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
        train_dataset=selected_dataset,
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)
