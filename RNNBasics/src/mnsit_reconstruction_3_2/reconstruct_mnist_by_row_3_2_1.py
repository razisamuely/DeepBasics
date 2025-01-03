import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import optuna
import argparse

from RNNBasics.src.models.lstm_autoencoder import LSTMAutoencoder
from RNNBasics.src.mnsit_reconstruction_3_2.training.trainer import train, evaluate
from RNNBasics.src.mnsit_reconstruction_3_2.utils.visualization import plot_reconstructions
from RNNBasics.src.mnsit_reconstruction_3_2.config import (
    DEVICE, BATCH_SIZE, N_EPOCHS, N_TRIALS, HIDDEN_SIZE_MIN, HIDDEN_SIZE_MAX,
    LEARNING_RATE_MIN, LEARNING_RATE_MAX, GRAD_CLIP_MIN, GRAD_CLIP_MAX,
    INPUT_SIZE, TRAIN_DATA_PATH, TEST_DATA_PATH, ARTIFACTS_DIR,
)

from RNNBasics.src.mnsit_reconstruction_3_2.utils.data_loader import get_dataloaders

def get_args():
    parser = argparse.ArgumentParser(description='LSTM Autoencoder for MNIST')
    
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                      help='batch size for training')
    parser.add_argument('--n-epochs', type=int, default=N_EPOCHS,
                      help='number of epochs to train')
    parser.add_argument('--n-trials', type=int, default=N_TRIALS,
                      help='number of Optuna trials')
    parser.add_argument('--device', type=str, default=DEVICE,
                      help='device to use for training')
    
    parser.add_argument('--hidden-size-min', type=int, default=HIDDEN_SIZE_MIN,
                      help='minimum hidden size for Optuna')
    parser.add_argument('--hidden-size-max', type=int, default=HIDDEN_SIZE_MAX,
                      help='maximum hidden size for Optuna')
    
    parser.add_argument('--lr-min', type=float, default=LEARNING_RATE_MIN,
                      help='minimum learning rate for Optuna')
    parser.add_argument('--lr-max', type=float, default=LEARNING_RATE_MAX,
                      help='maximum learning rate for Optuna')
    
    parser.add_argument('--grad-clip-min', type=float, default=GRAD_CLIP_MIN,
                      help='minimum gradient clipping value for Optuna')
    parser.add_argument('--grad-clip-max', type=float, default=GRAD_CLIP_MAX,
                      help='maximum gradient clipping value for Optuna')
    
    parser.add_argument('--artifacts-dir', type=str, default=ARTIFACTS_DIR,
                      help='directory to save artifacts')
    
    return parser.parse_args()

def objective(architectur_class, trial: optuna.Trial, train_loader: DataLoader, val_loader: DataLoader, 
             device: str, n_epochs: int, args) -> float:
    hidden_size = trial.suggest_int('hidden_size', args.hidden_size_min, args.hidden_size_max)
    learning_rate = trial.suggest_float('learning_rate', args.lr_min, args.lr_max, log=True)
    grad_clip = trial.suggest_float('grad_clip', args.grad_clip_min, args.grad_clip_max, log=True)
    
    model = architectur_class(input_size=INPUT_SIZE, hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    return train(model, train_loader, val_loader, optimizer, criterion,
                device, n_epochs, grad_clip, trial)

def main():
    args = get_args()
    os.makedirs(args.artifacts_dir, exist_ok=True)
    train_loader, test_loader = get_dataloaders(args.batch_size)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(LSTMAutoencoder,trial, train_loader, test_loader, 
                                         args.device, args.n_epochs, args),
                  n_trials=args.n_trials)
    
    best_params = study.best_params
    best_model = LSTMAutoencoder(input_size=INPUT_SIZE,
                                hidden_size=best_params['hidden_size']).to(args.device)
    
    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
    criterion = nn.MSELoss()
    
    _ = train(best_model, train_loader, test_loader, optimizer, criterion,
             args.device, args.n_epochs, best_params['grad_clip'])
    
    test_loss, originals, reconstructions = evaluate(best_model, test_loader, criterion, args.device)
    
    reconstruction_path = os.path.join(args.artifacts_dir, 'reconstructions_final_3_2_1.png')
    model_path = os.path.join(args.artifacts_dir, 'lstm_ae_mnist_3_2_1.pt')
    
    plot_reconstructions(originals, reconstructions, save_path=reconstruction_path)
    torch.save(best_model.state_dict(), model_path)


if __name__ == '__main__':
    main()