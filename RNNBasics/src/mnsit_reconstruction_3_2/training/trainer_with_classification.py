import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict
import optuna

class CombinedLoss:
    def __init__(self, reconstruction_weight=1.0, classification_weight=1.0):
        self.reconstruction_criterion = nn.MSELoss()
        self.classification_criterion = nn.CrossEntropyLoss()
        self.reconstruction_weight = reconstruction_weight
        self.classification_weight = classification_weight

    def __call__(self, reconstructed, class_probs, original, labels):
        recon_loss = self.reconstruction_criterion(reconstructed, original)
        class_loss = self.classification_criterion(class_probs, labels)
        total_loss = (self.reconstruction_weight * recon_loss + 
                     self.classification_weight * class_loss)
        return total_loss, recon_loss, class_loss

def compute_accuracy(predictions, targets):
    _, predicted_classes = torch.max(predictions, 1)
    return (predicted_classes == targets).float().mean().item()

def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_class_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    for batch, labels in dataloader:
        batch, labels = batch.to(device), labels.to(device)
        optimizer.zero_grad()

        reconstructed, class_probs = model(batch)
        total_loss_batch, recon_loss, class_loss = criterion(
            reconstructed, class_probs, batch, labels)

        total_loss_batch.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        accuracy = compute_accuracy(class_probs, labels)

        total_loss += total_loss_batch.item()
        total_recon_loss += recon_loss.item()
        total_class_loss += class_loss.item()
        total_accuracy += accuracy
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'class_loss': total_class_loss / num_batches,
        'accuracy': total_accuracy / num_batches
    }

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_class_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    originals = None
    reconstructions = None

    with torch.no_grad():
        for i, (batch, labels) in enumerate(dataloader):
            batch, labels = batch.to(device), labels.to(device)
            reconstructed, class_probs = model(batch)
            
            total_loss_batch, recon_loss, class_loss = criterion(
                reconstructed, class_probs, batch, labels)
            
            accuracy = compute_accuracy(class_probs, labels)

            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            total_class_loss += class_loss.item()
            total_accuracy += accuracy
            num_batches += 1

            if i == 0:
                originals = batch.cpu()
                reconstructions = reconstructed.cpu()

    metrics = {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'class_loss': total_class_loss / num_batches,
        'accuracy': total_accuracy / num_batches
    }

    return metrics, originals, reconstructions

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedLoss,
    device: str,
    n_epochs: int,
    grad_clip: float,
    trial: Optional[optuna.Trial] = None
) -> Dict[str, float]:
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_recon_loss': [], 'train_class_loss': [], 'train_accuracy': [],
        'val_loss': [], 'val_recon_loss': [], 'val_class_loss': [], 'val_accuracy': []
    }
    
    for epoch in range(n_epochs):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{n_epochs}:')
        print(f'Train - Loss: {train_metrics["loss"]:.6f}, '
              f'Recon Loss: {train_metrics["recon_loss"]:.6f}, '
              f'Class Loss: {train_metrics["class_loss"]:.6f}, '
              f'Accuracy: {train_metrics["accuracy"]*100:.2f}%')
        print(f'Val   - Loss: {val_metrics["loss"]:.6f}, '
              f'Recon Loss: {val_metrics["recon_loss"]:.6f}, '
              f'Class Loss: {val_metrics["class_loss"]:.6f}, '
              f'Accuracy: {val_metrics["accuracy"]*100:.2f}%')
        
        for key in train_metrics:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        
        if trial is not None:
            trial.report(val_metrics['loss'], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
    
    return best_val_loss, history