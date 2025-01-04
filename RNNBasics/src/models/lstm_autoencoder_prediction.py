import torch
import torch.nn as nn

class LSTMAutoencoderWithPrediction(nn.Module):
    """
    An LSTM Autoencoder with a single-layer encoder LSTM
    and a single-layer decoder LSTM. The same latent vector z
    (the last hidden state from encoder) is fed as input to the
    decoder at each time step.
    """

    def __init__(self, input_size, hidden_size, num_layers=1):
        """
        :param input_size:  Dimensionality of the input (e.g., number of features per time step).
        :param hidden_size: Dimensionality of the LSTM hidden state.
        :param num_layers:  Number of LSTM layers in encoder and decoder.
        """
        super(LSTMAutoencoderWithPrediction, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder LSTM: Takes (batch_size, seq_len, input_size)
        self.encoder_lstm = nn.LSTM(input_size, hidden_size,
                                    num_layers=num_layers, batch_first=True)

        # Decoder LSTM:
        # We feed the same z at each time step => input_size = hidden_size for the decoder.
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size,
                                    num_layers=num_layers, batch_first=True)

        # Linear layer to map decoder hidden states -> reconstruction
        self.output_layer = nn.Linear(hidden_size, input_size)

        # If we want a next-step prediction, add a separate layer for that
        self.next_step_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # ---- Encoder ----
        # We only need the final hidden state h_n (last layer).
        _, (h_n, c_n) = self.encoder_lstm(x)
        z = h_n[-1]  # shape: (batch_size, hidden_size)

        # Repeat z for each time step so the decoder sees it at every step.
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)  # (B, seq_len, hidden_size)

        # ---- Decoder ----
        decoder_outputs, _ = self.decoder_lstm(z_repeated)
        x_tilde = self.output_layer(decoder_outputs)  # shape: (batch_size, seq_len, input_size)

        last_hidden = decoder_outputs[:, -1, :]  # shape: (batch_size, hidden_size)
        y_hat = self.next_step_layer(last_hidden)  # shape: (batch_size, input_size)
        return x_tilde, y_hat
