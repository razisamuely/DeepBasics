from utils.train import *
import sys
sys.path.append('..')
from models.lstm_autoencoder import *
from config import *

def main():
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
        random_seed=42,
        optimizer=optimizer
    )

    print("\n=== Cross-Validation Results ===")
    for res in all_results:
        print(f"Split {res['split']}: "
              f"Train Loss = {res['train_loss']:.6f}, "
              f"Val Loss = {res['val_loss']:.6f}")
    print("\nBest validation loss across splits:",
          min(r['val_loss'] for r in all_results))

    all_symbols = np.array(sorted(full_df['symbol'].unique()))
    np.random.shuffle(all_symbols)
    cutoff = int(len(all_symbols) * 0.7)
    test_stocks = all_symbols[cutoff:]
    _, test_ds = get_train_test_datasets(full_df, all_symbols[:cutoff], test_stocks, seq_size=seq_size)
    # Now let's plot some reconstructions from the best model
    reconstruct_and_plot(best_model, test_ds, device=device, num_stocks=3)


if __name__ == "__main__":
    main()
