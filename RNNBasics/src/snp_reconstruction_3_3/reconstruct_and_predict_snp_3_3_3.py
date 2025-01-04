from utils.train import *
import sys
sys.path.append('..')
from models.lstm_autoencoder_prediction import *
import matplotlib.pyplot as plt
from config import *

def plot_cv_results(all_results):
    """
    Plots two graphs side-by-side:
      (1) Reconstruction train/val loss vs. epochs for each split
      (2) Prediction train/val loss vs. epochs for each split
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

    # 1) Reconstruction Loss
    for i, res in enumerate(all_results):
        epochs = range(1, len(res['train_loss']) + 1)
        axes[0].plot(epochs, res['train_loss'], marker='o', label=f"Train Split {res['split']}")
        axes[0].plot(epochs, res['val_loss'], marker='x', linestyle='--', label=f"Val Split {res['split']}")
        break

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Reconstruction Loss Across Epochs')
    axes[0].legend()

    # 2) Prediction Loss
    for i, res in enumerate(all_results):
        epochs = range(1, len(res['train_pred_loss']) + 1)
        axes[1].plot(epochs, res['train_pred_loss'], marker='o', label=f"Train Split {res['split']}")
        axes[1].plot(epochs, res['val_pred_loss'], marker='x', linestyle='--', label=f"Val Split {res['split']}")
        break

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Prediction Loss Across Epochs')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('../../artifacts/snp_3_3/train_val_losses_3_3_3.png')
    plt.close()
    # plt.show()


def main():

    # Read full CSV once
    full_df = pd.read_csv(data_path)

    # Perform cross-validation
    best_model, all_results = cross_validate_model(
        full_df,
        LSTMAutoencoderWithPrediction,
        n_splits=n_splits,
        seq_size=seq_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=learning_rate,
        hidden_size=hidden_size,
        num_layers=num_layers,
        device=device,
        random_seed=42,
        is_prediction=True
    )

    print("\n=== Cross-Validation Results ===")
    for res in all_results:
        print(f"Split {res['split']}: "
              f"Train Loss = {res['train_loss'][-1]:.6f}, "
              f"Val Loss = {res['val_loss'][-1]:.6f}, "
              f"Train Prediction Loss = {res['train_pred_loss'][-1]:.6f}, "
              f"Val Prediction Loss = {res['val_pred_loss'][-1]:.6f}")

    print("\nBest validation loss across splits:",
          min(r['val_loss'][-1] for r in all_results))

    plot_cv_results(all_results)

    torch.save(best_model.state_dict(), 'best_predicting_model_3_3_3.pt')


if __name__ == "__main__":
    main()
