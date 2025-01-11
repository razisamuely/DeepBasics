import matplotlib.pyplot as plt
import torch

def plot_and_save_examples(model, test_loader, device, n_examples, results_dir, name):
    """Plot and save reconstruction examples."""
    test_batch = next(iter(test_loader))
    x_in = test_batch[:n_examples].unsqueeze(-1).to(device)
    x_recon = model(x_in).cpu().detach().numpy()
    
    x_in_np = test_batch[:n_examples].numpy()
    x_recon_np = x_recon.squeeze(-1)
    
    for i in range(n_examples):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_in_np[i], label="Original", marker='o')
        ax.plot(x_recon_np[i], label="Reconstruction", marker='x')
        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Value")
        ax.set_title(f"Example #{i + 1} - Original vs. Reconstruction {name}")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(results_dir / f'reconstruction_example_{i+1}_{name}.png')
        plt.close()