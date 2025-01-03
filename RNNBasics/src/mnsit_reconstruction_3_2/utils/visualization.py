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



import matplotlib.pyplot as plt

def plot_training_history(history, save_dir, name = "training_history"):
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Total Loss')
    plt.plot(history['val_loss'], label='Val Total Loss')
    plt.plot(history['train_recon_loss'], label='Train Recon Loss')
    plt.plot(history['val_recon_loss'], label='Val Recon Loss')
    plt.plot(history['train_class_loss'], label='Train Class Loss')
    plt.plot(history['val_class_loss'], label='Val Class Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/{name}.png')
    plt.close()