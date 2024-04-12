import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from model import *
from dataset import *
from torch.utils.data import Dataset, DataLoader
from metrics import *
import tqdm

def load_data():
    with open("./data/the-verdict.txt", "r") as file:
        raw_text = file.read()

    return raw_text

if __name__ == '__main__':

    # Load data
    text = load_data()

    # Load tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create train and val datasets
    train_size = 0.9
    train_num = int(len(text)*0.9)

    train_text = text[:train_num]
    val_text = text[train_num:]

    # Create DataSets for trai and val
    train_dataset = GPTDataset(
        text = train_text,
        tokenizer = tokenizer,
        max_length = 126,
        stride = 1
    )

    val_dataset = GPTDataset(
        text = val_text,
        tokenizer = tokenizer,
        max_length = 126,
        stride = 1
    )

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = 2,
        shuffle = True,
        drop_last = True
    )

    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = 2,
        shuffle = False,
        drop_last = False
    )

    # Create iters
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    # Declare model arguments
    args = ModelArgs(
        emb_dim = 256,
        num_heads = 2,
        context_length = 256,
        vocab_size =tokenizer.n_vocab,
        num_blocks = 2
    )

    # Create model
    model = GPTModel(args)

    # Test model
    inputs_batch, target_batch = next(train_iter)
    model.eval()
    out = model(inputs_batch)

    # Declare some training hyperparams
    training_params = {
        'epochs': 10,
        'lr': 0.0001,
        'momentum': 0.9
    }

    # Create loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = training_params['lr'])

    # Now, begin training
    for epoch in tqdm.tqdm(range(training_params['epochs'])):
        epoch_loss = 0.0

        # Set model in training state
        model.train()
        for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(train_loader)):
            # Reset gradients
            optimizer.zero_grad()

            # Compute loss
            loss = calculate_batch_loss(model, inputs, targets)

            # Backward pass
            loss.backward()

            # Update params
            optimizer.step()

            # Update loss
            epoch_loss += loss.item()

        # Compute val loss
        model.eval()
        val_loss = calculate_loader_loss(model, val_loader)
        # Print epoch data
        print(f"Epoch: {epoch+1}, training loss: {epoch_loss / len(train_loader)}, val loss: {val_loss}")





    

    