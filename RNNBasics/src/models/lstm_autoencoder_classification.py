import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAutoencoderWithClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=10, num_layers=5):
        super(LSTMAutoencoderWithClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder_lstm = nn.LSTM(input_size, hidden_size, 
                                  num_layers=num_layers, batch_first=True)

        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, 
                                  num_layers=num_layers, batch_first=True)

        self.reconstruction_head = nn.Linear(hidden_size, input_size)

        self.classification_head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        _, (h_n, c_n) = self.encoder_lstm(x)
        z = h_n[-1] 

        class_logits = self.classification_head(z)
        class_probs = F.softmax(class_logits, dim=1)

        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_outputs, _ = self.decoder_lstm(z_repeated)
        reconstructed = self.reconstruction_head(decoder_outputs)

        return reconstructed, class_probs