from torch.utils.data import DataLoader
import copy
from torch import nn
from utils.dataset import *
from torch.optim.lr_scheduler import ExponentialLR

def cross_validate_model(
        df: pd.DataFrame,
        model_class,
        n_splits=5,
        seq_size=80,
        batch_size=16,
        num_epochs=10,
        lr=1e-3,
        hidden_size=32,
        num_layers=2,
        device='cpu',
        random_seed=42,
        optimizer='adam',
        is_prediction=False
):
    """
    Perform repeated random splits (Monte Carlo cross-validation) over stock symbols.
    1) Shuffle the list of unique symbols.
    2) Split them into a random train portion (~70%) and test portion (~30%).
    3) Train a new model on each split, measure validation loss.
    4) Keep track of the best model across all splits (lowest val loss).
    Returns:
        - best_model: Model instance (state_dict) with the best val performance
        - all_results: list of dicts containing { 'split': i, 'train_loss': X, 'val_loss': Y }
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Clean and prepare data
    df = df.dropna(subset=['high'])
    # Example of removing a known problematic symbol
    df = df[df['symbol'] != 'APTV']

    # All unique stocks
    all_symbols = np.array(sorted(df['symbol'].unique()))
    num_stocks = len(all_symbols)

    best_val_loss = float('inf')
    best_model_state = None

    all_results = []

    for split_idx in range(1, n_splits + 1):
        # Shuffle symbols
        np.random.shuffle(all_symbols)
        # 70/30 train/test
        cutoff = int(num_stocks * 0.7)
        train_stocks = all_symbols[:cutoff]
        test_stocks = all_symbols[cutoff:]

        # Build train/test Datasets
        train_ds, test_ds = get_train_test_datasets(df, train_stocks, test_stocks,
                                                    is_prediction=is_prediction, seq_size=seq_size)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Instantiate and train model
        model = model_class(input_size=1,
                            hidden_size=hidden_size,
                            num_layers=num_layers)

        print(f"\n=== Cross-Validation Split {split_idx}/{n_splits} ===")

        if train_loader.dataset.is_prediction:
            model, (train_recon_losses, train_pred_losses), (val_recon_losses, val_pred_losses) = train_model(
                model,
                train_loader,
                test_loader,
                num_epochs=num_epochs,
                lr=lr,
                device=device,
                optimizer=optimizer
            )

            final_val_loss = val_recon_losses[-1]

            # Keep track of results for this split
            all_results.append({
                'split': split_idx,
                'train_loss': train_recon_losses,
                'val_loss': val_recon_losses,
                'train_pred_loss': train_pred_losses,
                'val_pred_loss': val_pred_losses
            })

            # Check if this is the best so far
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_model_state = copy.deepcopy(model.state_dict())

        else:
            model, train_losses, val_losses = train_model(
                model,
                train_loader,
                test_loader,
                num_epochs=num_epochs,
                lr=lr,
                device=device
            )

            final_train_loss = train_losses[-1]
            final_val_loss = val_losses[-1]


            # Keep track of results for this split
            all_results.append({
                'split': split_idx,
                'train_loss': final_train_loss,
                'val_loss': final_val_loss
            })

            # Check if this is the best so far
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_model_state = copy.deepcopy(model.state_dict())

    # Re-load the best model weights into a fresh model instance
    best_model = model_class(input_size=1,
                             hidden_size=hidden_size,
                             num_layers=num_layers)
    best_model.load_state_dict(best_model_state)
    best_model.to(device)

    return best_model, all_results



def train_one_epoch(model, dataloader,
                    criterion,
                    optimizer, device='cpu'):
    """
    Train for one epoch. We can handle:
      - dataset returns only X (reconstruction only)
      - dataset returns (X, y) (reconstruction + prediction)
    """
    model.train()

    total_recon_loss = 0.0
    total_pred_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        # Figure out whether we have X only or (X, y)
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            X, y = batch
            X, y = X.to(device), y.to(device)
            x_hat, y_hat = model(X)   # model returns (x_hat, y_hat)
            recon_loss = criterion(x_hat, X)
            pred_loss = criterion(y_hat, y)
            loss = recon_loss + pred_loss

        else:
            # Only reconstruction data
            X = batch.to(device)
            # If your model always returns (x_hat, y_hat),
            # you can either ignore y_hat or have a separate model
            outputs = model(X)
            if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
                x_hat, _ = outputs
            else:
                # If your model only returns x_hat in this mode
                x_hat = outputs

            recon_loss = criterion(x_hat, X)
            pred_loss = torch.tensor(0.0, device=device)  # no prediction
            loss = recon_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumulate sums
        batch_size = X.size(0)
        total_samples += batch_size
        total_recon_loss += recon_loss.item() * batch_size
        total_pred_loss += pred_loss.item() * batch_size

    return total_recon_loss, total_pred_loss, total_samples


@torch.no_grad()
def evaluate(model, dataloader, criterion, device='cpu'):
    model.eval()

    total_recon_loss = 0.0
    total_pred_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            X, y = batch
            X, y = X.to(device), y.to(device)
            x_hat, y_hat = model(X)
            recon_loss = criterion(x_hat, X)
            pred_loss = criterion(y_hat, y)

        else:
            X = batch.to(device)
            outputs = model(X)
            if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
                x_hat, _ = outputs
            else:
                x_hat = outputs
            recon_loss = criterion(x_hat, X)
            pred_loss = torch.tensor(0.0, device=device)

        batch_size = X.size(0)
        total_samples += batch_size
        total_recon_loss += recon_loss.item() * batch_size
        total_pred_loss += pred_loss.item() * batch_size

    return total_recon_loss, total_pred_loss, total_samples


def train_model(model,
                train_loader: DataLoader,
                test_loader: DataLoader,
                num_epochs=10,
                lr=1e-3,
                device='cpu',
                optimizer='adam'):
    """
    Train and evaluate the model for a fixed train/test split.
    If the dataset provides only X (reconstruction only), pred_loss stays at 0.
    If the dataset provides (X, y), we get reconstruction + prediction losses.
    """
    criterion = nn.MSELoss()

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    model.to(device)

    train_recon_losses = []
    train_pred_losses = []
    val_recon_losses = []
    val_pred_losses = []

    for epoch in range(num_epochs):
        # TRAIN
        total_train_recon, total_train_pred, total_train_samples = train_one_epoch(
            model, train_loader,
            criterion,
            optimizer, device
        )
        avg_train_recon = total_train_recon / total_train_samples
        avg_train_pred = total_train_pred / total_train_samples

        # EVAL
        total_val_recon, total_val_pred, total_val_samples = evaluate(
            model, test_loader,
            criterion,
            device
        )
        avg_val_recon = total_val_recon / total_val_samples
        avg_val_pred = total_val_pred / total_val_samples

        train_recon_losses.append(avg_train_recon)
        train_pred_losses.append(avg_train_pred)
        val_recon_losses.append(avg_val_recon)
        val_pred_losses.append(avg_val_pred)

        if train_loader.dataset.is_prediction:
            print(f"[Epoch {epoch+1}/{num_epochs}] "
                  f"Train Recon: {avg_train_recon:.6f}, Train Pred: {avg_train_pred:.6f}, "
                  f"Val Recon: {avg_val_recon:.6f}, Val Pred: {avg_val_pred:.6f}")
        else:
            print(f"[Epoch {epoch+1}/{num_epochs}] "
                  f"Train Recon: {avg_train_recon:.6f}, "
                  f"Val Recon: {avg_val_recon:.6f}")


    if train_loader.dataset.is_prediction:
        return model, (train_recon_losses, train_pred_losses), (val_recon_losses, val_pred_losses)
    return model, train_recon_losses, val_recon_losses

