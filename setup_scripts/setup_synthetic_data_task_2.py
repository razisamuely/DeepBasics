import numpy as np
import os
import h5py

def generate_synthetic_sequences(n_sequences=10000, seq_length=50):
    sequences = np.random.random((n_sequences, seq_length))
    for j in range(n_sequences):
        i = np.random.randint(20, 31)
        start_idx = max(0, i - 5)
        end_idx = min(seq_length, i + 6)
        sequences[j, start_idx:end_idx] *= 0.1
    return sequences

def split_data(data, train_split=0.6, val_split=0.2):
    n_samples = len(data)
    train_idx = int(n_samples * train_split)
    val_idx = train_idx + int(n_samples * val_split)
    return data[:train_idx], data[train_idx:val_idx], data[val_idx:]

def save_data(train_data, val_data, test_data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'synthetic_sequences.h5')
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('train', data=train_data)
        f.create_dataset('validation', data=val_data)
        f.create_dataset('test', data=test_data)

def main():
    np.random.seed(42)
    sequences = generate_synthetic_sequences()
    train_data, val_data, test_data = split_data(sequences)
    output_dir = os.path.join('RNNBasics', 'data', 'synthetic_data')
    save_data(train_data, val_data, test_data, output_dir)
    print(f"Data saved successfully:")
    print(f"Training set shape: {train_data.shape}")
    print(f"Validation set shape: {val_data.shape}")
    print(f"Test set shape: {test_data.shape}")

if __name__ == '__main__':
    main()