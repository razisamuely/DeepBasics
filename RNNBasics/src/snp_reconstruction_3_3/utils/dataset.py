import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random

class SNPDataset(Dataset):
    """
    Dataset providing stock 'high' values for each distinct stock symbol.
    Sequences are truncated (or padded, if needed) to seq_size.
    """

    def __init__(self, df: pd.DataFrame, seq_size=80):
        self.seq_size = seq_size
        # Unique symbols to iterate over
        self.stocks = df['symbol'].unique()
        # Group the dataframe by symbol for faster indexing
        self.grouped = df.groupby('symbol')

    def __len__(self):
        # We have one "item" per unique stock symbol
        return len(self.stocks)

    def __getitem__(self, idx):
        """
        Return a tensor of shape (seq_size, 1) containing the 'high' values
        for the idx-th stock. If the number of data points is larger than seq_size,
        we truncate. If smaller, we could skip or pad. Here we’ll just truncate.
        """
        stock = self.stocks[idx]
        stock_df = self.grouped.get_group(stock).sort_values('date')  # sort by date if needed
        high_values = stock_df['high'].values

        max_start = len(high_values) - self.seq_size
        start_idx = random.randint(0, max_start)
        high_values = high_values[start_idx: start_idx + self.seq_size]

        high_values = (high_values - high_values.mean()) / high_values.std()

        return torch.tensor(high_values, dtype=torch.float).unsqueeze(-1).T
        # shape: (seq_size, 1), so LSTM can handle [batch, seq, feature]



def get_train_test_datasets(df: pd.DataFrame,
                            train_stocks: np.ndarray,
                            test_stocks: np.ndarray,
                            seq_size: int = 80):
    """
    Given a dataframe and two arrays of stock symbols (train_stocks, test_stocks),
    return SNPDataset objects for train and test sets.
    """
    train_df = df[df['symbol'].isin(train_stocks)]
    test_df = df[df['symbol'].isin(test_stocks)]

    train_ds = SNPDataset(train_df, seq_size=seq_size)
    test_ds = SNPDataset(test_df, seq_size=seq_size)

    return train_ds, test_ds




def reconstruct_and_plot(model, dataset, device='cpu', num_stocks=3):
    """
    Randomly select 'num_stocks' stocks from the dataset,
    reconstruct them using the trained model, and plot
    the original vs. reconstructed 'high' values.
    """
    model.eval()

    # Pick random indices from [0, len(dataset)-1]
    indices = np.random.choice(len(dataset), size=num_stocks, replace=False)

    fig, axs = plt.subplots(num_stocks, 1, figsize=(10, 4 * num_stocks))
    if num_stocks == 1:
        axs = [axs]

    for i, idx in enumerate(indices):
        with torch.no_grad():
            sequence = dataset[idx].unsqueeze(0).to(device)  # shape [1, seq_len, 1]
            output = model(sequence)
            # Convert to CPU numpy arrays
            original = sequence.squeeze().cpu().numpy()
            reconstructed = output.squeeze().cpu().numpy()


        # Plot
        axs[i].plot(original, label='Original', color='blue')
        axs[i].plot(reconstructed, label='Reconstructed', color='red', linestyle='--')
        axs[i].set_title(f"Stock index: {idx}")
        axs[i].legend()

    plt.tight_layout()
    plt.show()