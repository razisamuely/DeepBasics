import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import struct
import os
from typing import Tuple
import sys
import optuna

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(PROJECT_ROOT)

from RNNBasics.src.models.lstm_autoencoder import LSTMAutoencoder
from RNNBasics.src.mnsit_reconstruction_3_2.training.trainer import train, evaluate
from RNNBasics.src.mnsit_reconstruction_3_2.utils.visualization import plot_reconstructions
from RNNBasics.src.mnsit_reconstruction_3_2.config import (
    DEVICE, BATCH_SIZE, N_EPOCHS, N_TRIALS, HIDDEN_SIZE_MIN, HIDDEN_SIZE_MAX,
    LEARNING_RATE_MIN, LEARNING_RATE_MAX, GRAD_CLIP_MIN, GRAD_CLIP_MAX,
    INPUT_SIZE, TRAIN_DATA_PATH, TEST_DATA_PATH, ARTIFACTS_DIR,
    MODEL_SAVE_PATH, RECONSTRUCTIONS_SAVE_PATH
)

class MNISTSequenceDataset(Dataset):
    def __init__(self, images_file: str):
        with open(images_file, 'rb') as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            self.images = np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows, cols)
        self.images = self.images.astype(np.float32) / 255.0
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.images[idx], dtype=torch.float32)

def get_dataloaders(batch_size: int = BATCH_SIZE) -> Tuple[DataLoader, DataLoader]:
    train_ds = MNISTSequenceDataset(TRAIN_DATA_PATH)
    test_ds = MNISTSequenceDataset(TEST_DATA_PATH)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def objective(trial: optuna.Trial, train_loader: DataLoader, val_loader: DataLoader, 
             device: str, n_epochs: int) -> float:
    hidden_size = trial.suggest_int('hidden_size', HIDDEN_SIZE_MIN, HIDDEN_SIZE_MAX)
    learning_rate = trial.suggest_float('learning_rate', LEARNING_RATE_MIN, LEARNING_RATE_MAX, log=True)
    grad_clip = trial.suggest_float('grad_clip', GRAD_CLIP_MIN, GRAD_CLIP_MAX, log=True)
    
    model = LSTMAutoencoder(input_size=INPUT_SIZE, hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    return train(model, train_loader, val_loader, optimizer, criterion,
                device, n_epochs, grad_clip, trial)

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    train_loader, test_loader = get_dataloaders()
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_loader, test_loader, 
                                         DEVICE, N_EPOCHS),
                  n_trials=N_TRIALS)
    
    best_params = study.best_params
    best_model = LSTMAutoencoder(input_size=INPUT_SIZE,
                                hidden_size=best_params['hidden_size']).to(DEVICE)
    
    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
    criterion = nn.MSELoss()
    
    _ = train(best_model, train_loader, test_loader, optimizer, criterion,
             DEVICE, N_EPOCHS, best_params['grad_clip'])
    
    test_loss, originals, reconstructions = evaluate(best_model, test_loader, criterion, DEVICE)
    print(f'Final Test Loss: {test_loss:.6f}')
    
    plot_reconstructions(originals, reconstructions, save_path=RECONSTRUCTIONS_SAVE_PATH)
    torch.save(best_model.state_dict(), MODEL_SAVE_PATH)
    
    print("\nBest trial:")
    print(f"  Value: {study.best_trial.value:.6f}")
    print("  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

if __name__ == '__main__':
    main()