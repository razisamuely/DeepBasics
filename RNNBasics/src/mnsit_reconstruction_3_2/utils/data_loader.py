import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import struct
from typing import Tuple
from RNNBasics.src.mnsit_reconstruction_3_2.config import (
    BATCH_SIZE, TRAIN_DATA_PATH, TEST_DATA_PATH,
    TRAIN_LABELS_PATH, TEST_LABELS_PATH, VAL_SPLIT
)

class MNISTSequenceDataset(Dataset):
    def __init__(self, images_file: str, labels_file: str = None, is_pixel_sequence: bool = False):
        with open(images_file, 'rb') as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            self.images = np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows, cols)

        if is_pixel_sequence:
            self.images = self.images.reshape(size, -1, 1)

        self.images = self.images.astype(np.float32) / 255.0
        
        self.labels = None
        if labels_file is not None:
            with open(labels_file, 'rb') as f:
                magic, size = struct.unpack(">II", f.read(8))
                self.labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        return image

def get_dataloaders(batch_size: int = BATCH_SIZE,
                    train_data_path: str = TRAIN_DATA_PATH,
                    test_data_path: str = TEST_DATA_PATH) -> Tuple[DataLoader, DataLoader]:
    train_ds = MNISTSequenceDataset(train_data_path)
    test_ds = MNISTSequenceDataset(test_data_path)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def get_dataloaders_with_labels(
    batch_size: int = BATCH_SIZE,
    train_data_path: str = TRAIN_DATA_PATH,
    train_labels_path: str = TRAIN_LABELS_PATH,
    test_data_path: str = TEST_DATA_PATH,
    test_labels_path: str = TEST_LABELS_PATH,
    val_split: float = VAL_SPLIT,
    is_pixel_sequence: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    # Create full training dataset
    full_train_ds = MNISTSequenceDataset(train_data_path, train_labels_path, is_pixel_sequence)
    test_ds = MNISTSequenceDataset(test_data_path, test_labels_path, is_pixel_sequence)
    
    # Split training into train and validation
    train_size = int((1 - val_split) * len(full_train_ds))
    val_size = len(full_train_ds) - train_size
    
    train_ds, val_ds = torch.utils.data.random_split(
        full_train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader