from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
from utils.dataset import *
sys.path.append('..')
from models.lstm_autoencoder_prediction import *
from config import *
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':

    df = pd.read_csv(data_path)
    all_symbols = df['symbol'].unique()

    np.random.shuffle(all_symbols)
    # 70/30 train/test
    cutoff = int(len(all_symbols) * 0.7)
    train_stocks = all_symbols[:cutoff]
    test_stocks = all_symbols[cutoff:]

    model = LSTMAutoencoderWithPrediction(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(torch.load('best_predicting_model_3_3_3.pt', weights_only=True))
    model.eval()

    _, dataset = get_train_test_datasets(df, train_stocks, test_stocks,
                                                is_prediction=True, seq_size=seq_size)

    # Configuration
    batch_size = 1  # Assuming sequence is processed one at a time for simplicity
    sequence_length = len(dataset)  # Assuming dataset provides entire test sequence
    T = seq_size  # Total length of the sequence

    # Load the test data
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Get the test sequence (assume the dataset provides a tensor of shape [T, feature_dim])
    x_test, y_test = next(iter(test_loader))

    # Split the test sequence
    T_half = T // 2
    input_sequence = x_test[:, :T_half, :]  # First half of the test sequence

    actual_future_sequence = x_test[:, T_half:, :]  # Ground truth for the second half

    # Prepare the model
    model.eval()  # Set model to evaluation mode

    # Perform multi-step predictions
    predicted_sequence = []
    current_input = input_sequence.clone()

    with torch.no_grad():
        for t in range(T_half):
            _, next_prediction = model(current_input)
            next_prediction = next_prediction.unsqueeze(0)  # Shape: [1, 1, 1]
            predicted_sequence.append(next_prediction)

            # Shift the existing values in `current_input` and insert `next_prediction`
            current_input[:, :-1, :] = current_input[:, 1:, :].clone()  # Shift left with clone
            current_input[:, -1, :] = next_prediction[:, 0, :]  # Insert the new prediction at the last position

    # Convert predictions to a tensor
    predicted_sequence = torch.stack(predicted_sequence).squeeze(-1).squeeze(-1).unsqueeze(0)

    print(predicted_sequence.shape, actual_future_sequence.shape)
    # Evaluate the multi-step prediction
    mse_loss = torch.nn.MSELoss()
    prediction_error = mse_loss(predicted_sequence, actual_future_sequence)

    print("Multi-Step Prediction Error (MSE):", prediction_error.item())

    # Visualization
    time = range(T)
    plt.plot(np.array(time[:T_half]), input_sequence.numpy()[0, :, 0], label="Input Sequence")
    plt.plot(np.array(time[T_half:]), actual_future_sequence.numpy()[0, :, 0], label="Actual Future Sequence")
    plt.plot(np.array(time[T_half:]), predicted_sequence.numpy()[0, :, 0], label="Predicted Future Sequence", linestyle="dashed")
    plt.legend()
    plt.title("Multi-Step Prediction")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.savefig('../../artifacts/snp_3_3/predict_next_steps_3_3_4.png')
    plt.close()
