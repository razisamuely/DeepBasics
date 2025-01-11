import torch
import torch.nn as nn
from typing import Optional
import optuna
import  tqdm
import loguru
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
    """Training function for both regular training and optuna trials."""
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in tqdm.trange(n_epochs):
        model.train()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip)
        val_loss = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # log last n epochs loss
        if epoch % 10 == 0:
            train_loss = sum(train_losses[-10:]) / 10
            val_loss = sum(val_losses[-10:]) / 10
            loguru.logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        
    
    return train_losses, val_losses

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