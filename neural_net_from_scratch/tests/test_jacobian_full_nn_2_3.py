

def build_full_network(L, input_dim, output_dim, hidden_dim, activation="tanh"):
    """
    Build a full network with L layers.
    For simplicity:
    - First layer: input_dim -> hidden_dim
    - (L-2) hidden layers: hidden_dim -> hidden_dim
    - Last layer: hidden_dim -> output_dim
    """
    layer_configs = []
    # First layer
    layer_configs = [
        {"type": "layer", "input_dim": input_dim, "output_dim": hidden_dim, "activation": "tanh"},
        {"type": "layer", "input_dim": hidden_dim, "output_dim": hidden_dim, "activation": "tanh"},
        {"type": "layer", "input_dim": hidden_dim, "output_dim": output_dim, "activation": 'tanh'},
    ]

    network = DynamicNeuralNetwork(layer_configs)
    return network



def gradient_test_full_network(L=3, input_dim=10, hidden_dim=20, output_dim=5, epsilon=0.2):
    # Generate some random data and labels
    X = np.random.randn(input_dim, 1)
    y = np.random.randn(output_dim, 1)

    # creating random one hot vector
    y[y == y.max()], y[~(y == y.max())] = 1, 0

    d = np.random.randn(input_dim, 1)

    # Build the network
    network = build_full_network(L, input_dim, output_dim, hidden_dim, activation="tanh")

    # d = d / np.linalg.norm(d)  # Normalize direction

    linear_errors = []
    quadratic_errors = []
    eps_values = []

    sfml = SoftmaxLoss()


    for i in range(20):
        eps = epsilon * (0.5 ** i)
        eps_values.append(eps)

        x_forward, x_eps_forward = network.forward(X), network.forward(X + eps * d)

        x_forward, x_eps_forward = sfml.sofotmax(x_forward), sfml.sofotmax(x_eps_forward)

        forward_loss, eps_forward_loss = sfml.loss(x_forward, y), sfml.loss(x_eps_forward, y)

        linear_error = abs(forward_loss - eps_forward_loss)

        current_gradient = sfml.loss_gradient(x_forward, y)
        for layer in reversed(network.layers):
            current_gradient = layer.backward(current_gradient)

        quadratic_error = abs(forward_loss - eps_forward_loss + eps * (d.T @ current_gradient)[0][0])


        linear_errors.append(linear_error)
        quadratic_errors.append(quadratic_error)


    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.loglog(eps_values, linear_errors, 'b.-', label='Linear Error')
    plt.loglog(eps_values, quadratic_errors, 'r.-', label='Quadratic Error')

    ref_eps = np.array([eps_values[0], eps_values[-1]])
    ref_linear = ref_eps * linear_errors[0] / eps_values[0]
    ref_quad = ref_eps ** 2 * quadratic_errors[0] / eps_values[0] ** 2
    plt.loglog(ref_eps, ref_linear, 'b--', alpha=0.5, label='O(ε)')
    plt.loglog(ref_eps, ref_quad, 'r--', alpha=0.5, label='O(ε²)')

    plt.grid(True)
    plt.xlabel('epsilon')
    plt.ylabel('Error')
    plt.title('Full Network Gradient Test (L = {})'.format(L))
    plt.legend()
    # plt.savefig(f'full_network_gradient_test_L{L}.png')
    # plt.close()
    plt.show()

    return linear_errors, quadratic_errors


if __name__ == "__main__":

    import sys
    import os
    import matplotlib.pyplot as plt

    # just for testing, import activation class and relu for activation function and run a test on synthetic data
    current_dir = os.path.dirname(__file__)
    activations_path = os.path.abspath(os.path.join(current_dir, '../../'))
    sys.path.append(activations_path)

    import numpy as np
    import matplotlib.pyplot as plt
    from neural_net_from_scratch.src.models.neural_network import DynamicNeuralNetwork
    from neural_net_from_scratch.src.losses.sofmax_loss import *

    # Example usage: test a network with L=3 layers
    linear_errors, quadratic_errors = gradient_test_full_network(L=3)
