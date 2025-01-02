import torch
import matplotlib.pyplot as plt

def plot_reconstructions(originals: torch.Tensor, 
                        reconstructions: torch.Tensor,
                        n_images: int = 10,
                        save_path: str = 'mnist_reconstructions.png'):
    """Plot original vs reconstructed images."""
    plt.figure(figsize=(20, 4))
    
    for i in range(n_images):
        # Original
        plt.subplot(2, n_images, i + 1)
        plt.imshow(originals[i], cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        # Reconstruction
        plt.subplot(2, n_images, n_images + i + 1)
        plt.imshow(reconstructions[i], cmap='gray')
        plt.title('Reconstructed')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()