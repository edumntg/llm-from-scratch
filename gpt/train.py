import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from model import *
from dataset import *
from torch.utils.data import Dataset, DataLoader
from metrics import *
import tqdm
import argparse
import os
import urllib.request
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():

    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    text_data = None
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    return text_data

if __name__ == '__main__':
    print("Connected to device:", device)

    parser = argparse.ArgumentParser(
        description="LLM training parameters",
    )

    parser.add_argument('--data', type=str, default='./data/the-verdict.txt', help="Text file containing sample data")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs to train the model")
    parser.add_argument('--print_every', type=int, default=1000, help="Number of iterations between printing sample outputs")
    parser.add_argument('--emb_dim', default=256, type=int, help="Default embedding dimension")
    parser.add_argument('--n_heads', default=12, type=int, help="Number of heads on multi-head attention block")
    parser.add_argument('--n_blocks', type=int, default=12, help="Number of transfer blocks")
    parser.add_argument('--context_length', type=int, default=256, help="Number of dimensions for context vectors")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for train/val data")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate for training loop")
    parser.add_argument('--wd', type=float, default=0.01, help="Weight decay used uring training (for AdamW optimizer only)")
    parser.add_argument('--optimizer', type=str, default="adam", help="Optimizer to be used during training")
    parser.add_argument('--drop_rate', type=float, default=0.1, help="Drop rate for model")
    cargs = parser.parse_args()

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
        batch_size = cargs.batch_size,
        shuffle = True,
        drop_last = True
    )

    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = cargs.batch_size,
        shuffle = False,
        drop_last = False
    )

    # Create iters
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    # Declare model arguments
    args = ModelArgs(
        emb_dim = cargs.emb_dim,
        num_heads = cargs.n_heads,
        context_length = cargs.context_length,
        vocab_size =tokenizer.n_vocab,
        num_blocks = cargs.n_blocks,
        dropout=cargs.drop_rate
    )

    # Create model
    model = GPTModel(args)
    model.to(device)

    # Test model
    inputs_batch, target_batch = next(train_iter)
    inputs_batch = inputs_batch.to(device)
    out = model(inputs_batch)
    print(out.shape)

    # Declare some training hyperparams
    training_params = {
        'epochs': cargs.epochs,
        'lr': cargs.lr,
        'wd': cargs.wd
    }

    # Create loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = training_params['lr'])
    if cargs.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr = training_params['lr'], weight_decay=training_params['wd'])

    # Now, begin training
    steps = 0
    for epoch in tqdm.tqdm(range(training_params['epochs'])):
        epoch_loss = 0.0

        # Set model in training state
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Reset gradients
            optimizer.zero_grad()

            # Compute loss
            loss = calculate_batch_loss(model, inputs, targets, device)

            # Backward pass
            loss.backward()

            # Update params
            optimizer.step()

            # Update loss
            epoch_loss += loss.item()

            steps += 1
            if steps % cargs.print_every == 0:
                # Compute while training/val loss
                train_loss = calculate_loader_loss(model, train_loader, device)
                model.eval()
                val_loss = calculate_loader_loss(model, val_loader, device)
                model.train()
                print(f"Epoch: {epoch+1} Step: {steps} - Train Loss: {train_loss}, Val Loss: {val_loss}")

        # Save model
        torch.save(model.state_dict(), "./trained_model.pt")


        # Compute val loss
        # model.eval()
        # val_loss = calculate_loader_loss(model, val_loader)
        # # Print epoch data
        # print(f"Epoch: {epoch+1}, training loss: {epoch_loss / len(train_loader)}, val loss: {val_loss}")





    

    