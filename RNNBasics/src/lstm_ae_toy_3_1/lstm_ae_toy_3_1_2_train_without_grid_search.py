import argparse
import torch

from RNNBasics.src.lstm_ae_toy_3_1.config import (
    DEFAULT_BATCH_SIZE, DEFAULT_N_EPOCHS, DEFAULT_DEVICE, 
    DEFAULT_HIDDEN_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_GRAD_CLIP,
    DEFAULT_OPTIMIZER, DEFAULT_LOSS, INPUT_SIZE, 
    N_EXAMPLES_TO_PLOT, RESULTS_DIR, AVAILABLE_OPTIMIZERS, AVAILABLE_LOSSES
)

from RNNBasics.src.lstm_ae_toy_3_1.training.trainer import train, evaluate
from RNNBasics.src.lstm_ae_toy_3_1.utils.model_utils import get_optimizer, get_loss_function, save_model_results
from RNNBasics.src.lstm_ae_toy_3_1.utils.visualization import plot_and_save_examples
from RNNBasics.src.models.lstm_autoencoder import LSTMAutoencoder
from RNNBasics.src.synthetic_data.synthetic_data_2 import get_dataloaders
import loguru

def get_args():
    parser = argparse.ArgumentParser(description='LSTM Autoencoder Training')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--n-epochs', type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE)
    
    # Model parameters
    parser.add_argument('--hidden-size', type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--grad-clip', type=float, default=DEFAULT_GRAD_CLIP)
    
    # Optimizer and loss parameters
    parser.add_argument('--optimizer', type=str, default=DEFAULT_OPTIMIZER,
                      choices=AVAILABLE_OPTIMIZERS)
    parser.add_argument('--loss', type=str, default=DEFAULT_LOSS,
                      choices=AVAILABLE_LOSSES)
    
    return parser.parse_args()

def main():
    args = get_args()
    
    loguru.logger.info(f"Training LSTM Autoencoder with the following parameters: {args}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    loguru.logger.info("Data loaded successfully")
    
    # Initialize model
    model = LSTMAutoencoder(
        input_size=INPUT_SIZE,
        hidden_size=args.hidden_size
    ).to(args.device)
    loguru.logger.info("Model initialized successfully")
    
    # Setup optimizer and loss function
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.learning_rate)
    loguru.logger.info("Optimizer initialized successfully")

    criterion = get_loss_function(args.loss)
    loguru.logger.info("Loss function initialized successfully")

    # Train the model
    train_losses, val_losses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=args.device,
        n_epochs=args.n_epochs,
        grad_clip=args.grad_clip
    )
    loguru.logger.info("Training completed successfully")
    
    # Evaluate on test set
    test_loss = evaluate(model, test_loader, criterion, args.device)
    loguru.logger.info(f"Test loss: {test_loss}")
    
    
    plot_and_save_examples(model, test_loader, args.device, N_EXAMPLES_TO_PLOT, RESULTS_DIR,name ="test_set")
    plot_and_save_examples(model, train_loader, args.device, N_EXAMPLES_TO_PLOT, RESULTS_DIR,name ="train_set")

    def plot_val_vs_train_loss(train_losses, val_losses):
        import matplotlib.pyplot as plt
        plt.plot(train_losses, label='train_loss')
        plt.plot(val_losses, label='val_loss')
        plt.legend()
        plt.savefig(RESULTS_DIR / 'loss_plot_f.png')
        plt.close()
        
    plot_val_vs_train_loss(train_losses[10:], val_losses[10:])

    # save model and results
    torch.save(model.state_dict(), RESULTS_DIR / '3_1_model.pth')

if __name__ == "__main__":
    main()