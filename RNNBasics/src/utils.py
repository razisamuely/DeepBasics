import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Train an LSTM Autoencoder.")
    parser.add_argument("--input_size", type=int, default=1, help="Dimensionality of the input features.")
    parser.add_argument("--hidden_size", type=int, default=64, help="Dimensionality of the LSTM hidden state.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"], help="Optimizer type.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")

    # You can add more application-specific parameters if desired.
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        # batch has shape (batch_size, seq_len)
        # we need shape (batch_size, seq_len, input_size); here input_size=1
        x = batch.unsqueeze(-1).to(device)  # (batch_size, seq_len, 1)

        optimizer.zero_grad()
        x_recon = model(x)

        loss = criterion(x_recon, x)
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            x = batch.unsqueeze(-1).to(device)  # (batch_size, seq_len, 1)
            x_recon = model(x)
            loss = criterion(x_recon, x)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)


def plot_sequences(x_list, title="Sequence Samples"):
    """
    Utility to plot a list of 1D sequences in separate subplots.
    x_list: list of 1D arrays (or something convertible to numpy).
    """
    fig, axes = plt.subplots(len(x_list), 1, figsize=(6, 3 * len(x_list)))
    if len(x_list) == 1:
        axes = [axes]  # make it iterable

    for i, x_seq in enumerate(x_list):
        axes[i].plot(x_seq, marker='o', linestyle='-')
        axes[i].set_xlabel("Time (t)")
        axes[i].set_ylabel("Value")
        axes[i].set_title(f"{title} #{i + 1}")
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()
