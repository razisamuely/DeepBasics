
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import struct
import os
from typing import Tuple, List
import sys
from typing import Optional
import optuna
import tqdm
def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    n_epochs: int,
    grad_clip: float,
    trial: Optional[optuna.Trial] = None
) -> float:
    """Training function adapted for MNIST."""
    best_val_loss = float('inf')
    
    for epoch in tqdm.tqdm(range(n_epochs), desc="Epochs"):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{n_epochs}:')
        print(f'Train Loss: {train_loss:.6f}')
        print(f'Val Loss: {val_loss:.6f}')
        
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss

def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0

    for batch in tqdm.tqdm(dataloader, desc="Processing Batches"):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        reconstructed = model(batch)
        
        loss = criterion(reconstructed, batch)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model: nn.Module, 
            test_loader: DataLoader,
            criterion: nn.Module,
            device: str) -> Tuple[float, List[torch.Tensor], List[torch.Tensor]]:

    model.eval()
    total_loss = 0.0
    originals = []
    reconstructions = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            total_loss += loss.item()
            
            if i == 0:
                originals = batch.cpu()
                reconstructions = reconstructed.cpu()
    
    return total_loss / len(test_loader), originals, reconstructions