from utils.train import *
import sys
sys.path.append('..')
from models.lstm_autoencoder import *


def main():
    # Hyperparameters
    data_path = "/Users/amitaharoni/PycharmProjects/Amit Python/University Courses/Deep Learning/DeepBasics/RNNBasics/data/SP 500 Stock Prices 2014-2017.csv"  # <-- replace with your actual CSV path
    seq_size = 80
    batch_size = 64
    hidden_size = 64
    num_layers = 1
    learning_rate = 1e-2
    num_epochs = 200
    n_splits = 3  # how many random splits you want
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Read full CSV once
    full_df = pd.read_csv(data_path)

    # Perform cross-validation
    best_model, all_results = cross_validate_model(
        full_df,
        LSTMAutoencoder,
        n_splits=n_splits,
        seq_size=seq_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=learning_rate,
        hidden_size=hidden_size,
        num_layers=num_layers,
        device=device,
        random_seed=42
    )

    print("\n=== Cross-Validation Results ===")
    for res in all_results:
        print(f"Split {res['split']}: "
              f"Train Loss = {res['train_loss']:.6f}, "
              f"Val Loss = {res['val_loss']:.6f}")
    print("\nBest validation loss across splits:",
          min(r['val_loss'] for r in all_results))

    # Now best_model holds the best-performing model from cross-validation.
    # If we want to do reconstructions, we must pick or recreate a test dataset:
    # As an example, re-creating the last random split used:
    # (But typically you’d do a final train/test with a chosen split
    #  or a final evaluation on unseen data.)
    # We'll just do it for demonstration using the final split from cross_validate_model.

    # Last random split
    # (In practice, you might want to store the train/test stocks for the best split,
    #  or pick the last one, or re-run a final splitted dataset.)
    # Here, let's just pick the final all_symbols from that loop.
    # A better approach is to store them inside cross_validate_model if you want to do more analysis.
    # For demonstration, let's build a random test dataset from scratch:

    # Re-shuffle
    all_symbols = np.array(sorted(full_df['symbol'].unique()))
    np.random.shuffle(all_symbols)
    cutoff = int(len(all_symbols) * 0.7)
    test_stocks = all_symbols[cutoff:]
    _, test_ds = get_train_test_datasets(full_df, all_symbols[:cutoff], test_stocks, seq_size)
    # Now let's plot some reconstructions from the best model
    reconstruct_and_plot(best_model, test_ds, device=device, num_stocks=3)


if __name__ == "__main__":
    main()
