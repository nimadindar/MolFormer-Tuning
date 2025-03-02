import torch
import torch.nn.functional as F

from tqdm import tqdm



def train_model(model, tokenizer, train_dataloader, test_dataloader, num_epochs):
    """
    Trains a transformer-based model for regression using Mean Squared Error (MSE) loss.

    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to be trained.
    tokenizer : Callable
        A tokenizer function that processes SMILES strings into model-compatible inputs.
    train_dataloader : DataLoader
        A PyTorch DataLoader providing batches of (SMILES, label) pairs for training.
    test_dataloader : DataLoader
        A PyTorch DataLoader providing batches of (SMILES, label) pairs for validation.
    num_epochs : int
        The number of training epochs.

    Returns:
    --------
    None
        The function trains the model and saves it as "regression_head_w_extra_data.pth".
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define learning rate and optimizer
    lr = 0.020104429120603076
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []  # Stores training loss for each epoch
    val_losses = []    # Stores validation loss for each epoch

    # Training loop over multiple epochs
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        # Iterate through training data
        for smile, label in progress_bar:
            label = label.to(device).float()

            smiles_token = tokenizer(smile, padding=True, truncation=True, return_tensors="pt")
            smiles_token = {k: v.to(device) for k, v in smiles_token.items()}

            optimizer.zero_grad()
            output = model(smiles_token).squeeze()

            loss = F.mse_loss(output, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Compute average training loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        epoch_losses.append(avg_epoch_loss)

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for smile, label in test_dataloader:
                label = label.to(device).float()

                smiles_token = tokenizer(smile, padding=True, truncation=True, return_tensors="pt")
                smiles_token = {k: v.to(device) for k, v in smiles_token.items()}

                output = model(smiles_token).squeeze()

                loss = F.mse_loss(output, label)
                val_loss += loss.item()

        # Compute average validation loss for the epoch
        avg_val_loss = val_loss / len(test_dataloader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_epoch_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

    # Save the trained model
    torch.save(model, "../weights/regression_head_w_extra_data.pth")

    return epoch_losses, val_losses