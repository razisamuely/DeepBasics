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

    def __init__(self, df: pd.DataFrame, seq_size=80, is_prediction=False):
        self.seq_size = seq_size
        self.stocks = df['symbol'].unique()
        self.grouped = df.groupby('symbol')
        self.is_prediction = is_prediction

    def __len__(self):
        return len(self.stocks)

    def __getitem__(self, idx):
        stock = self.stocks[idx]
        stock_df = self.grouped.get_group(stock).sort_values('date')
        high_values = stock_df['high'].values

        # Example logic for picking a start index
        max_start = len(high_values) - self.seq_size - (1 if self.is_prediction else 0)
        start_idx = random.randint(0, max_start)

        # Take seq_size points
        high_values_sample = high_values[start_idx: start_idx + self.seq_size]
        mean_ = high_values_sample.mean()
        std_ = high_values_sample.std()

        # Normalize
        high_values_sample_normalized = (high_values_sample - mean_) / std_

        X = torch.tensor(
            high_values_sample_normalized, dtype=torch.float
        ).unsqueeze(-1).T

        if self.is_prediction:
            # Next value after the sequence
            next_value = high_values[start_idx + self.seq_size]
            next_value_normalized = (next_value - mean_) / std_

            # Return (X, y)
            y = torch.tensor(next_value_normalized, dtype=torch.float).unsqueeze(-1)
            return X, y
        else:
            # Return only X
            return X



def get_train_test_datasets(df: pd.DataFrame,
                            train_stocks: np.ndarray,
                            test_stocks: np.ndarray,
                            is_prediction: bool = False,
                            seq_size: int = 80):
    """
    Given a dataframe and two arrays of stock symbols (train_stocks, test_stocks),
    return SNPDataset objects for train and test sets.
    """
    train_df = df[df['symbol'].isin(train_stocks)]
    test_df = df[df['symbol'].isin(test_stocks)]

    train_ds = SNPDataset(train_df, seq_size=seq_size, is_prediction=is_prediction)
    test_ds = SNPDataset(test_df, seq_size=seq_size, is_prediction=is_prediction)

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