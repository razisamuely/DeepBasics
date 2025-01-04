import torch

data_path = "../../data/SP 500 Stock Prices 2014-2017.csv"
seq_size = 80
batch_size = 16
hidden_size = 64
num_layers = 2
learning_rate = 1e-2
num_epochs = 100
n_splits = 3  # how many random splits you want
device = 'cuda' if torch.cuda.is_available() else 'cpu'
