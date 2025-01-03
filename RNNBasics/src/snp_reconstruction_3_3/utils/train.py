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
        train_ds, test_ds = get_train_test_datasets(df, train_stocks, test_stocks, seq_size)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Instantiate and train model
        model = model_class(input_size=seq_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers)

        print(f"\n=== Cross-Validation Split {split_idx}/{n_splits} ===")
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
    best_model = model_class(input_size=seq_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers)
    best_model.load_state_dict(best_model_state)
    best_model.to(device)

    return best_model, all_results



def train_one_epoch(model, dataloader, criterion, optimizer, device='cpu'):
    """
    Train the model for one epoch on the given dataloader.
    Returns the total loss (sum, not average).
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        batch = batch.to(device)  # (B, seq_size, 1)
        optimizer.zero_grad()

        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.size(0)

    return total_loss


def evaluate(model, dataloader, criterion, device='cpu'):
    """
    Evaluate the model on the given dataloader.
    Returns the total loss (sum, not average).
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs, batch)
            total_loss += loss.item() * batch.size(0)

    return total_loss


def train_model(model,
                train_loader: DataLoader,
                test_loader: DataLoader,
                num_epochs=10,
                lr=1e-3,
                device='cpu'):
    """
    Train and evaluate the LSTM AE model for a fixed train/test split.
    Returns the trained model, plus the train & val losses for each epoch.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.999)
    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        total_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        total_val_loss = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_val_loss = total_val_loss / len(test_loader.dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"[Epoch {epoch + 1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    return model, train_losses, val_losses

