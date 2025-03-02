import torch
import torch.nn.functional as F

from tqdm import tqdm

def LiSSA(model, tokenizer, data_loader, vec, damp, repeat, depth, scale):
    """
    LiSSA estimates the inverse Hessian-vector product (IHVP)
    using a recursive approach. This is useful in influence function calculations to approximate how small
    perturbations in training data affect model predictions.

    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to evaluate.
    tokenizer : Callable
        A tokenizer function that processes SMILES strings into model-compatible inputs.
    data_loader : DataLoader
        A PyTorch DataLoader providing batches of (SMILES, label) pairs for model evaluation.
    vec : torch.Tensor
        The vector to be multiplied by the inverse Hessian matrix.
    damp : float
        A damping factor to stabilize the Hessian approximation.
    repeat : int
        The number of times to repeat the LiSSA recursion for averaging.
    depth : int
        The number of iterations per recursion (truncation depth).
    scale : float
        A scaling factor to control the approximation step size.

    Returns:
    --------
    torch.Tensor
        The estimated inverse Hessian-vector product (IHVP), averaged over the repetitions.
    """

    # Determine the device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move model and vector to the appropriate device
    model.to(device)
    vec = vec.to(device)

    # Initialize the inverse Hessian-vector product estimate as a zero tensor
    ihvp = torch.zeros_like(vec, device=device)

    # Repeat the Hessian-vector product estimation multiple times for averaging
    for r in range(repeat):
        # Initialize h_est with the input vector
        h_est = vec.clone()

        # Iterate through the dataset for a limited number of steps (depth)
        for t, (smiles, label) in enumerate(data_loader):
            if t >= depth:
                break  # Stop when reaching the truncation depth

            # Move labels to the appropriate device
            label = label.to(device).float()

            # Tokenize SMILES strings into model input format
            smiles_token = tokenizer(smiles, padding=True, truncation=True, return_tensors="pt")
            smiles_token = {k: v.to(device) for k, v in smiles_token.items()}

            # Zero out gradients before computing new ones
            model.zero_grad()

            # Forward pass: compute model predictions
            outputs = model(smiles_token).squeeze()

            # Compute Mean Squared Error (MSE) loss
            loss = F.mse_loss(outputs, label.squeeze())

            # Compute gradients of the loss w.r.t. model parameters
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)

            # Ensure None gradients are replaced with zero tensors (to avoid errors in computation)
            grads = [g if g is not None else torch.zeros_like(p, requires_grad=True) for g, p in zip(grads, model.parameters())]

            # Flatten gradients into a single vector
            flat_grads = torch.cat([g.view(-1) for g in grads])

            # Compute Hessian-vector product (HVP) using implicit differentiation
            hvp = torch.autograd.grad(flat_grads, model.parameters(), grad_outputs=h_est, retain_graph=True, allow_unused=True)
            hvp = [g if g is not None else torch.zeros_like(p, requires_grad=True) for g, p in zip(hvp, model.parameters())]

            # Flatten HVP into a single vector
            hvp = torch.cat([h.view(-1) for h in hvp])

            # Update h_est using the recursive LiSSA formula
            with torch.no_grad():
                hvp = hvp + damp * h_est  # Apply damping for numerical stability
                h_est = vec + h_est - hvp / scale  # Update h_est with the computed HVP

        # Accumulate the estimated IHVP over multiple repetitions
        ihvp += h_est / scale

    # Return the averaged IHVP over all repetitions
    return ihvp / repeat


def compute_influence(extra_data, data_loader, tokenizer, model, damp, repeat, depth, scale):
    """
    Computes the influence score of each data point in `data_loader` by estimating the effect
    of removing that point on the model’s loss function. This is achieved using influence functions 
    and the LiSSA algorithm to approximate the inverse Hessian-vector product (IHVP).

    Parameters:
    -----------
    extra_data : pandas.DataFrame
        A DataFrame containing SMILES strings and other metadata where influence scores will be stored.
    data_loader : DataLoader
        A PyTorch DataLoader providing batches of (SMILES, label) pairs for model evaluation.
    tokenizer : Callable
        A tokenizer function that processes SMILES strings into model-compatible inputs.
    model : torch.nn.Module
        The neural network model for which influence scores are computed.
    damp : float
        A damping factor for stabilizing the Hessian approximation.
    repeat : int
        The number of times to repeat the LiSSA recursion for averaging.
    depth : int
        The number of iterations per recursion (truncation depth).
    scale : float
        A scaling factor to control the approximation step size.

    Returns:
    --------
    None
        The function modifies `extra_data` by adding an 'influence_score' column and saving it to a CSV file.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Initialize influence scores in the extra_data DataFrame
    extra_data['influence_score'] = 0.0

    # Iterate through the dataset to compute influence scores
    for smiles, label in tqdm(data_loader, desc="Computing Influence Scores"):
        smiles = smiles[0]  # Ensure SMILES is a string rather than a list

        label = label.to(device).float()
        smiles_token = tokenizer(smiles, padding=True, truncation=True, return_tensors="pt")
        smiles_token = {k: v.to(device) for k, v in smiles_token.items()}

        model.zero_grad()
        outputs = model(smiles_token).squeeze()
        loss = F.mse_loss(outputs, label.squeeze())

        # Compute gradients of the loss with respect to model parameters
        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=False, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, model.parameters())]

        # Flatten gradients into a single vector
        flat_grads = torch.cat([g.view(-1) for g in grads])

        # Compute the inverse Hessian-vector product (IHVP) using LiSSA
        ihvp = LiSSA(model, tokenizer, data_loader, flat_grads, damp, repeat, depth, scale)

        # Compute the influence score as the negative dot product of IHVP and the gradient
        influence_score = -torch.dot(ihvp, flat_grads).item()

        # Update the DataFrame with the computed influence score
        extra_data.loc[extra_data['SMILES'].astype(str) == str(smiles), 'influence_score'] = influence_score

    # Save the updated DataFrame to a CSV file
    extra_data.to_csv("../datasets/updated_data_w_influence_scores.csv", index=False)