import torch

def calculate_batch_loss(model, inputs_batch, targets_batch, device = "cpu"):

    inputs_batch, targets_batch = inputs_batch.to(device), targets_batch.to(device)
    # First, compute logits
    logits = model(inputs_batch)

    # Flatten first two dimensions
    logits = logits.flatten(0, 1)

    # Compute loss
    loss = torch.nn.functional.cross_entropy(logits, targets_batch.flatten())

    return loss

def calculate_loader_loss(model, data_loader):
    total_loss = 0.0
    for i, (inputs_batch, targets_batch) in enumerate(data_loader):
        total_loss += calculate_batch_loss(model, inputs_batch, targets_batch).item()
    
    total_loss /= len(data_loader)

    return total_loss