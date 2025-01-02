import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
import h5py
import os

class SyntheticSequenceDataset(Dataset):
    """
    Synthetic dataset of sequences in [0,1], each of length 50.
    Each sequence has a 'dip' region where values are multiplied by 0.1.
    """

    def __init__(self, num_samples=10000, seq_len=50, split='train'):
        super().__init__()

        np.random.seed(42)
        random.seed(42)

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


def plot_sample_sequences(dataset, num_samples=5):
    """Plot a few sample sequences from the dataset"""
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        seq = dataset[i].numpy()
        plt.plot(seq, label=f'Sequence {i+1}')
    
    plt.title('Sample Sequences from Synthetic Dataset')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('sample_sequences.png')
    plt.close()


def save_dataset(save_path='RNNBasics/data/synthetic_data/synthetic_sequences.h5'):
    """Save all splits of the dataset to an HDF5 file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create datasets for each split
    train_ds = SyntheticSequenceDataset(split='train')
    val_ds = SyntheticSequenceDataset(split='val')
    test_ds = SyntheticSequenceDataset(split='test')
    
    # Save to HDF5
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('train', data=train_ds.data.numpy())
        f.create_dataset('val', data=val_ds.data.numpy())
        f.create_dataset('test', data=test_ds.data.numpy())


if __name__ == '__main__':
    # Create and plot sample sequences
    train_ds = SyntheticSequenceDataset(split='train')
    plot_sample_sequences(train_ds)
    
    # Save the dataset
    save_dataset()
    
    print("Dataset has been saved and sample sequences have been plotted.")
    
    # Verify the saved data
    with h5py.File('RNNBasics/data/synthetic_data/synthetic_sequences.h5', 'r') as f:
        print("\nDataset splits and shapes:")
        for key in f.keys():
            print(f"{key}: {f[key].shape}")
        
        # Create a figure with subplots for each split
        plt.figure(figsize=(15, 10))
        
        for idx, key in enumerate(['train', 'val', 'test']):
            plt.subplot(3, 1, idx + 1)
            
            # Plot first 5 sequences from each split
            data = f[key][:]
            for i in range(5):
                plt.plot(data[i], label=f'Sequence {i+1}')
            
            plt.title(f'{key.capitalize()} Split - Sample Sequences')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('h5_sequences.png')
        plt.close()