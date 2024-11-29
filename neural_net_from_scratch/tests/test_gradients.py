from neural_net_from_scratch.src.utils.loss_functions import SoftmaxLoss
import numpy as np
import matplotlib.pyplot as plt

EPSILON = 1e-5
N_TESTS = 8
N_SAMPLES = 100
N_FEATURES = 5
N_CLASSES = 3
INIT_SCALE = 0.01
PLOT_FILE_PATH = 'neural_net_from_scratch/artifacts/gradient_test.png'
def compute_gradient_errors(sfml, X, W, y, d, epsilon):
    """Compute zero and first order errors for gradient testing"""
    y_one_hot = sfml.one_hot(y, N_CLASSES)
    net_output = sfml.forward(X, W)
    loss_0 = sfml.cross_entropy_loss(net_output, y_one_hot)
    
    zero_order = np.zeros(N_TESTS)
    first_order = np.zeros(N_TESTS)

    print("\nGradient Test for Augmented Weights")
    print("k\tZero-order\tFirst-order\tRatio")
    print("-" * 50)

    for k in range(N_TESTS):
        eps_k = epsilon * (0.5 ** k)
        
        W_perturbed = W + eps_k * d
        net_output_perturbed = sfml.forward(X, W_perturbed)
        loss_k = sfml.cross_entropy_loss(net_output_perturbed, y_one_hot)

        zero_order[k] = abs(loss_k - loss_0)
        first_order[k] = abs(loss_k - (loss_0 + eps_k * np.sum(sfml.gradients(X, y, net_output, W) * d)))

        ratio = "" if k == 0 else f"{first_order[k]/first_order[k-1]:.3f}"
        print(f"{k+1}\t{zero_order[k]:.2e}\t{first_order[k]:.2e}\t{ratio}")

    return zero_order, first_order

def plot_errors(zero_order, first_order):
    plt.figure(figsize=(8, 6))
    plt.semilogy(range(1, N_TESTS + 1), zero_order, 'o-', label='Zero-order')
    plt.semilogy(range(1, N_TESTS + 1), first_order, 'o-', label='First-order')
    plt.grid(True)
    plt.legend()
    plt.title('Gradient Test - Augmented Weights')
    plt.xlabel('k')
    plt.ylabel('Error (log scale)')
    plt.tight_layout()
    plt.savefig(PLOT_FILE_PATH)

def test_gradients():
    W = np.random.randn(N_FEATURES + 1, N_CLASSES) * INIT_SCALE
    X = np.random.randn(N_SAMPLES, N_FEATURES)
    y = np.random.randint(0, N_CLASSES, size=N_SAMPLES)
    sfml = SoftmaxLoss()

    d = np.random.randn(*W.shape)
    d = d / np.linalg.norm(d)

    zero_order, first_order = compute_gradient_errors(sfml, X, W, y, d, EPSILON)
    
    plot_errors(zero_order, first_order)

if __name__ == "__main__":
    test_gradients()