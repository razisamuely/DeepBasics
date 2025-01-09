import matplotlib.pyplot as plt
import h5py


if __name__ == '__main__':
    # Verify the saved data
    with h5py.File('../../data/synthetic_data/synthetic_sequences.h5', 'r') as f:
        print("\nDataset splits and shapes:")
        for key in f.keys():
            print(f"{key}: {f[key].shape}")
        
        # Create a figure with subplots for each split
        plt.figure(figsize=(15, 10))
        
        for idx, key in enumerate(['train', 'validation', 'test']):
            plt.subplot(3, 1, idx + 1)
            
            # Plot first 5 sequences from each split
            data = f[key][:]
            for i in range(5):
                plt.plot(data[i], label=f'Sequence {i+1}')
            
            plt.title(f'{key.capitalize()} Split - Sample Sequences')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('../../artifacts/synthetic_data_3_1_1.png')
        plt.close()