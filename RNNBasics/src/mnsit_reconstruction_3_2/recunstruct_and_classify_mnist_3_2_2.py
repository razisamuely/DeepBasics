import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import optuna
import argparse

from RNNBasics.src.models.lstm_autoencoder_classification import LSTMAutoencoderWithClassifier
from RNNBasics.src.mnsit_reconstruction_3_2.training.trainer_with_classification import train, evaluate, CombinedLoss
from RNNBasics.src.mnsit_reconstruction_3_2.utils.visualization import (
    plot_reconstructions, plot_training_history
)
from RNNBasics.src.mnsit_reconstruction_3_2.config import (
    DEVICE, BATCH_SIZE, N_EPOCHS, N_TRIALS, HIDDEN_SIZE_MIN, HIDDEN_SIZE_MAX,
    LEARNING_RATE_MIN, LEARNING_RATE_MAX, GRAD_CLIP_MIN, GRAD_CLIP_MAX,
    INPUT_SIZE, ARTIFACTS_DIR
)
from RNNBasics.src.mnsit_reconstruction_3_2.utils.data_loader import get_dataloaders_with_labels

def get_args():
    parser = argparse.ArgumentParser(description='LSTM Autoencoder with Classification for MNIST')
    
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--n-epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--n-trials', type=int, default=N_TRIALS)
    parser.add_argument('--device', type=str, default=DEVICE)
    
    parser.add_argument('--hidden-size-min', type=int, default=HIDDEN_SIZE_MIN)
    parser.add_argument('--hidden-size-max', type=int, default=HIDDEN_SIZE_MAX)
    parser.add_argument('--lr-min', type=float, default=LEARNING_RATE_MIN)
    parser.add_argument('--lr-max', type=float, default=LEARNING_RATE_MAX)
    parser.add_argument('--grad-clip-min', type=float, default=GRAD_CLIP_MIN)
    parser.add_argument('--grad-clip-max', type=float, default=GRAD_CLIP_MAX)
    
    parser.add_argument('--recon-weight', type=float, default=1.0,
                      help='weight for reconstruction loss')
    parser.add_argument('--class-weight', type=float, default=1.0,
                      help='weight for classification loss')
    
    parser.add_argument('--artifacts-dir', type=str, 
                      default=os.path.join(ARTIFACTS_DIR, 'mnist_3_2_2'))
    
    return parser.parse_args()

def objective(trial: optuna.Trial, train_loader: DataLoader, val_loader: DataLoader, 
             device: str, n_epochs: int, args) -> float:
    hidden_size = trial.suggest_int('hidden_size', args.hidden_size_min, args.hidden_size_max)
    learning_rate = trial.suggest_float('learning_rate', args.lr_min, args.lr_max, log=True)
    grad_clip = trial.suggest_float('grad_clip', args.grad_clip_min, args.grad_clip_max, log=True)
    
    model = LSTMAutoencoderWithClassifier(
        input_size=INPUT_SIZE, 
        hidden_size=hidden_size
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CombinedLoss(
        reconstruction_weight=args.recon_weight,
        classification_weight=args.class_weight
    )
    
    val_loss, _ = train(model, train_loader, val_loader, optimizer, criterion,
                     device, n_epochs, grad_clip, trial)
    return val_loss

def main():
    args = get_args()
    os.makedirs(args.artifacts_dir, exist_ok=True)
    
    train_loader, val_loader, test_loader = get_dataloaders_with_labels(args.batch_size)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, 
                                         args.device, args.n_epochs, args),
                  n_trials=args.n_trials)
    
    best_params = study.best_params
    best_model = LSTMAutoencoderWithClassifier(
        input_size=INPUT_SIZE,
        hidden_size=best_params['hidden_size']
    ).to(args.device)
    
    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
    criterion = CombinedLoss(
        reconstruction_weight=args.recon_weight,
        classification_weight=args.class_weight
    )
    
    # Train final model and get history
    _, history = train(
        best_model, train_loader, val_loader, optimizer, criterion,
        args.device, args.n_epochs, best_params['grad_clip']
    )
    
    # Final evaluation
    test_metrics, originals, reconstructions = evaluate(
        best_model, test_loader, criterion, args.device
    )
    
    print("\nTest Set Results:")
    print(f"Loss: {test_metrics['loss']:.6f}")
    print(f"Reconstruction Loss: {test_metrics['recon_loss']:.6f}")
    print(f"Classification Loss: {test_metrics['class_loss']:.6f}")
    print(f"Accuracy: {test_metrics['accuracy']*100:.2f}%")
    
    # Save artifacts
    plot_reconstructions(
        originals, 
        reconstructions,
        save_path=os.path.join(args.artifacts_dir, 'reconstructions_final_3_2_1.png')
    )
    
    plot_training_history(
        history,
        save_dir=args.artifacts_dir,
        name='training_history_3_2_1.png'
    )
    
    torch.save(best_model.state_dict(), 
              os.path.join(args.artifacts_dir, 'lstm_ae_classifier.pt'))
    
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    print(f"\nAll artifacts saved to: {args.artifacts_dir}")

if __name__ == '__main__':
    main()