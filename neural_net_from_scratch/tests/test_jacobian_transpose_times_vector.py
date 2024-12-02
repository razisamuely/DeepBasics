import numpy as np
import sys
import os

# just for testing, import activation class and relu for activation function and run a test on synthetic data
activations_path = os.path.abspath('../src/activations')
sys.path.append(activations_path)
models_path = os.path.abspath('../src/models')
sys.path.append(models_path)

from activations_class import *
from relu import *
from tanh import *
from neural_network import *

def gradient_check(nn, x, y, epsilon=1e-7):
    # Store original weights
    original_weights = [w.copy() for w in nn.weights]

    # Initialize numerical gradients
    numerical_grads = []
    for i, W in enumerate(nn.weights):
        num_grad = np.zeros_like(W)
        for idx in np.ndindex(W.shape):
            original_value = W[idx]

            # Compute f(W + epsilon)
            W[idx] = original_value + epsilon
            plus_loss = 0.5 * np.mean((nn.forward(x) - y) ** 2)

            # Compute f(W - epsilon)
            W[idx] = original_value - epsilon
            minus_loss = 0.5 * np.mean((nn.forward(x) - y) ** 2)

            # Restore original value
            W[idx] = original_value

            # Compute numerical gradient
            num_grad[idx] = (plus_loss - minus_loss) / (2 * epsilon)

        numerical_grads.append(num_grad)

    # Restore original weights
    nn.weights = original_weights

    # Recompute analytical gradients with restored weights
    nn.forward(x)
    analytical_grads = nn.backward(x, y)

    # Compare gradients
    for idx, (a_grad, n_grad) in enumerate(zip(analytical_grads, numerical_grads)):
        numerator = np.linalg.norm(a_grad - n_grad)
        denominator = np.linalg.norm(a_grad) + np.linalg.norm(n_grad) + 1e-8
        diff = numerator / denominator
        print(f"Layer {idx + 1} relative difference: {diff}")


def test_jacobian_transpose_times_vector():
    # input and output
    N = 30
    n_features = 2
    out_dim = 1
    X = np.random.randn(n_features, N)
    X.sort()
    y = (2 * (np.arange(N) + np.random.randn(N) * 0.001) / N) - 1
    relu_activation = ActivationFunction(activation=tanh, derivative=tanh_derivative)



    # define the network
    nn = NeuralNetwork(
        layer_sizes=[n_features, 5, 10, out_dim],
        activation_function=relu_activation,
        last_activation_function=None
    )

    gradient_check(nn, X, y)


if __name__ == '__main__':
    test_jacobian_transpose_times_vector()