import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random


class SyntheticSequenceDataset(Dataset):
    """
    Synthetic dataset of sequences in [0,1], each of length 50.
    Each sequence has a 'dip' region where values are multiplied by 0.1.
    """

    def __init__(self, num_samples=10000, seq_len=50, split='train'):
        super().__init__()

        # Fix random seed for reproducibility (optional)
        np.random.seed(42)
        random.seed(42)

        # Generate raw data in [0, 1]
        data = np.random.rand(num_samples, seq_len)

        # Post-process each sequence
        for j in range(num_samples):
            i = random.randint(20, 30)  # i in [20, 30]
            start = max(0, i - 5)
            end = min(seq_len, i + 5)
            data[j, start:end] *= 0.1

        # Split indices: 60% train, 20% valid, 20% test
        train_end = int(0.6 * num_samples)
        valid_end = int(0.8 * num_samples)

        if split == 'train':
            self.data = data[:train_end]
        elif split == 'val':
            self.data = data[train_end:valid_end]
        elif split == 'test':
            self.data = data[valid_end:]
        else:
            raise ValueError("split must be one of {'train','val','test'}")

        # Convert to PyTorch tensor (float32)
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Shape will be (seq_len,)
        x = self.data[index]
        return x


def get_dataloaders(batch_size=128):
    train_ds = SyntheticSequenceDataset(split='train')
    val_ds = SyntheticSequenceDataset(split='val')
    test_ds = SyntheticSequenceDataset(split='test')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
