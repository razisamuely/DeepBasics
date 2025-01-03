import torch.nn as nn

class LSTMAutoencoder(nn.Module):
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
        super(LSTMAutoencoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder LSTM: Takes (batch_size, seq_len, input_size)
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

        # Decoder LSTM: Takes (batch_size, seq_len, input_size=hidden_size for z)
        # However, we feed the same z at each time step, so effectively input_size = hidden_size
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)

        # Final layer to map hidden state to reconstruction of original input size
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        """
        Forward pass of the LSTM AE.

        :param x: Input of shape (batch_size, seq_len, input_size).
        :return: Reconstruction x_tilde of the same shape as x.
        """
        batch_size, seq_len, _ = x.size()

        # ---- Encoder ----
        # Encoder LSTM returns outputs (all hidden states), (h_n, c_n).
        # h_n, c_n have shape (num_layers, batch_size, hidden_size).
        _, (h_n, c_n) = self.encoder_lstm(x)

        # The latent vector z is the last hidden state of the encoder
        # But we will reshape it to (batch_size, hidden_size) for convenience
        z = h_n[-1]  # shape: (batch_size, hidden_size)

        # We want to feed z at each time step to the decoder. So replicate it seq_len times.
        # This gives shape (batch_size, seq_len, hidden_size).
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)

        # ---- Decoder ----
        # We'll feed the repeated z to the decoder LSTM.
        decoder_outputs, _ = self.decoder_lstm(z_repeated)

        # Map each decoder hidden state to the input dimension
        x_tilde = self.output_layer(decoder_outputs)  # shape: (batch_size, seq_len, input_size)

        return x_tilde

    def encode(self, x):
        """
        Encode input x to the latent space.

        :param x: Input of shape (batch_size, seq_len, input_size).
        :return: Latent vector z of shape (batch_size, hidden_size).
        """
        _, (h_n, c_n) = self.encoder_lstm(x)
        z = h_n[-1]
        return z
    
