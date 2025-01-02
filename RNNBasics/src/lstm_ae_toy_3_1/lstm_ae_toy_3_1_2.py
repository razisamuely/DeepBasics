import argparse
import optuna

from RNNBasics.src.lstm_ae_toy_3_1.config import (DEFAULT_BATCH_SIZE, DEFAULT_N_EPOCHS, DEFAULT_N_TRIALS, 
                                                  DEFAULT_DEVICE, DEFAULT_HIDDEN_SIZE_MIN, DEFAULT_HIDDEN_SIZE_MAX, 
                                                  DEFAULT_LR_MIN, DEFAULT_LR_MAX, DEFAULT_GRAD_CLIP_MIN, 
                                                  DEFAULT_GRAD_CLIP_MAX, DEFAULT_OPTIMIZER, DEFAULT_LOSS, 
                                                  AVAILABLE_OPTIMIZERS, AVAILABLE_LOSSES, INPUT_SIZE, 
                                                  N_EXAMPLES_TO_PLOT, RESULTS_DIR)

from RNNBasics.src.lstm_ae_toy_3_1.training.trainer import train, evaluate
from RNNBasics.src.lstm_ae_toy_3_1.utils.model_utils import get_optimizer, get_loss_function, save_model_results
from RNNBasics.src.lstm_ae_toy_3_1.utils.visualization import plot_and_save_examples
from RNNBasics.src.models.lstm_autoencoder import LSTMAutoencoder
from RNNBasics.src.synthetic_data.synthetic_data_2 import get_dataloaders

def get_args():
    parser = argparse.ArgumentParser(description='LSTM Autoencoder Training')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--n-epochs', type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument('--n-trials', type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE)
    
    # Grid search ranges
    parser.add_argument('--hidden-size-min', type=int, default=DEFAULT_HIDDEN_SIZE_MIN)
    parser.add_argument('--hidden-size-max', type=int, default=DEFAULT_HIDDEN_SIZE_MAX)
    parser.add_argument('--lr-min', type=float, default=DEFAULT_LR_MIN)
    parser.add_argument('--lr-max', type=float, default=DEFAULT_LR_MAX)
    parser.add_argument('--grad-clip-min', type=float, default=DEFAULT_GRAD_CLIP_MIN)
    parser.add_argument('--grad-clip-max', type=float, default=DEFAULT_GRAD_CLIP_MAX)
    
    # Optimizer and loss parameters
    parser.add_argument('--optimizer', type=str, default=DEFAULT_OPTIMIZER,
                      choices=AVAILABLE_OPTIMIZERS)
    parser.add_argument('--loss', type=str, default=DEFAULT_LOSS,
                      choices=AVAILABLE_LOSSES)
    
    return parser.parse_args()

def objective(trial, train_loader, val_loader, device, n_epochs, args):
    hidden_size = trial.suggest_int('hidden_size', args.hidden_size_min, args.hidden_size_max)
    learning_rate = trial.suggest_float('learning_rate', args.lr_min, args.lr_max, log=True)
    grad_clip = trial.suggest_float('grad_clip', args.grad_clip_min, args.grad_clip_max)
    
    model = LSTMAutoencoder(input_size=INPUT_SIZE, hidden_size=hidden_size).to(device)
    optimizer = get_optimizer(args.optimizer, model.parameters(), learning_rate)
    criterion = get_loss_function(args.loss)
    
    return train(model, train_loader, val_loader, optimizer, criterion, 
                device, n_epochs, grad_clip, trial)

def main():
    args = get_args()
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, 
                                         args.device, args.n_epochs, args),
                  n_trials=args.n_trials)
    
    best_model = LSTMAutoencoder(
        input_size=INPUT_SIZE,
        hidden_size=study.best_params['hidden_size']
        
    ).to(args.device)
    
    optimizer = get_optimizer(args.optimizer, best_model.parameters(), 
                            study.best_params['learning_rate'])
    criterion = get_loss_function(args.loss)
    
    train(best_model, train_loader, val_loader, optimizer, criterion,
          args.device, args.n_epochs, study.best_params['grad_clip'])
    
    test_loss = evaluate(best_model, test_loader, criterion, args.device)
    
    save_model_results(study, best_model, test_loss, args, RESULTS_DIR)
    plot_and_save_examples(best_model, test_loader, args.device, 
                          N_EXAMPLES_TO_PLOT, RESULTS_DIR)

if __name__ == "__main__":
    main()