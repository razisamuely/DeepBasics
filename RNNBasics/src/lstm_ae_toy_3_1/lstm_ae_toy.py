import numpy as np
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

import sys
import os
current_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(src_path)

from utils import *
from models.lstm_autoencoder import *
from synthetic_data.synthetic_data_2 import *



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)

    sample_batch = next(iter(train_loader))
    sample_batch_np = sample_batch[:3].numpy()
    plot_sequences(sample_batch_np, title="Synthetic Data Examples")


    # parameters for grid search
    hidden_sizes = [32, 64, 128]  # e.g., smaller than seq_len=50
    learning_rates = [1e-3, 1e-4]
    grad_clips = [0.1, 1.0, 5.0]

    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_config = None
    best_model_state = None

    for hs in hidden_sizes:
        for lr in learning_rates:
            for gc in grad_clips:
                print(f"\n=== Training LSTM AE with hidden_size={hs}, lr={lr}, grad_clip={gc} ===")

                model = LSTMAutoencoder(input_size=1, hidden_size=hs).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)

                n_epochs = 15
                for epoch in range(n_epochs):
                    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip=gc)
                    val_loss = evaluate(model, val_loader, criterion, device)

                    print(f"Epoch [{epoch + 1}/{n_epochs}]  "
                          f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_config = (hs, lr, gc)
                    best_model_state = model.state_dict()

    print(f"\nBest Config: hidden_size={best_config[0]}, lr={best_config[1]}, grad_clip={best_config[2]}")
    print(f"Best Val Loss: {best_val_loss:.4f}")

    best_model = LSTMAutoencoder(input_size=1, hidden_size=best_config[0]).to(device)
    best_model.load_state_dict(best_model_state)

    test_loss = evaluate(best_model, test_loader, criterion, device)
    print(f"Test Loss (best model): {test_loss:.4f}")

    # we'll take a small batch from test_loader and plot a couple of examples
    test_batch = next(iter(test_loader))
    x_in = test_batch[:2].unsqueeze(-1).to(device)
    x_recon = best_model(x_in).cpu().detach().numpy()

    x_in_np = test_batch[:2].numpy()
    x_recon_np = x_recon.squeeze(-1)

    # original and reconstruction side by side
    for i in range(2):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_in_np[i], label="Original", marker='o')
        ax.plot(x_recon_np[i], label="Reconstruction", marker='x')
        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Value")
        ax.set_title(f"Example #{i + 1} - Original vs. Reconstruction")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
