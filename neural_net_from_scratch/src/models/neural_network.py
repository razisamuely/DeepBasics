import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, activation_function=None, last_activation_function=None):
        """
        Initialize a neural network with augmented input for bias.

        Args:
            layer_sizes (list): List of integers representing the size of each layer.
            activation_function (callable): Activation function to use.
            last_activation_function (callable): Last activation function to use.
        """
        self.layer_sizes = layer_sizes
        self.activation_function = activation_function
        self.last_activation_function = last_activation_function
        self.weights = []
        self.outputs = []  # To store intermediate outputs during forward pass
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize weights for all layers."""
        for i in range(len(self.layer_sizes) - 1):
            # +1 for augmented bias input in the current layer
            self.weights.append(
                np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i] + 1) * np.sqrt(2 / (self.layer_sizes[i]))
            )

    def forward(self, X):
        """
        Perform a forward pass through the network.

        Args:
            X (ndarray): Input data of shape (n_features, n_samples).

        Returns:
            ndarray: Output of the network.
        """

        self.outputs = []  # Clear previous outputs
        self.pre_activations = []  # To store pre-activation values
        X = np.vstack([X, np.ones(X.shape[1])])  # Augment input with bias
        self.outputs.append(X)

        for i, W in enumerate(self.weights):
            Z = W @ X  # Pre-activation value
            self.pre_activations.append(Z)
            if i != len(self.weights) - 1:
                if self.activation_function is not None:
                    X = self.activation_function.activation(Z)
                else:
                    X = Z
                X = np.vstack([X, np.ones(X.shape[1])])  # Add bias for the next layer
            else:
                X = Z
            self.outputs.append(X)

        if self.last_activation_function is not None:
            X = self.last_activation_function.activation(X)
            self.outputs[-1] = X  # Replace the last output with the final activated output
        return X

    def backward(self, x, y):
        """
        Backward pass through the network.

        Args:
            x (np.ndarray): Input data.
            y (np.ndarray): True labels.

        Returns:
            list: Gradients for each weight matrix in the network.
        """
        gradients = []
        # Compute the error at the output layer
        final_output = self.outputs[-1]
        error = final_output - y  # Gradient of loss wrt output
        if self.last_activation_function:
            error *= self.last_activation_function.derivative(self.pre_activations[-1])  # Use pre-activation

        # Backpropagation through each layer
        for i in reversed(range(len(self.weights))):
            output = self.outputs[i]
            grad_w = error @ output.T / x.shape[1]
            gradients.insert(0, grad_w)

            if i > 0:
                W_no_bias = self.weights[i][:, :-1]  # Exclude the bias term
                error = W_no_bias.T @ error
                if self.activation_function:
                    error *= self.activation_function.derivative(self.pre_activations[i - 1])  # Use pre-activation

        self.outputs = []
        self.pre_activations = []
        return gradients

    def update_weights(self, gradients, learning_rate):
        """
        Update weights using the computed gradients.

        Args:
            gradients (list): Gradients for each weight matrix.
            learning_rate (float): Learning rate for gradient descent.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients[i]

    def gradient_check(self, x, y, epsilon=1e-7):
        # Store original weights
        original_weights = [w.copy() for w in self.weights]

        # Compute analytical gradients
        self.forward(x)
        analytical_grads = self.backward(x, y)

        # Initialize numerical gradients
        numerical_grads = []
        for i, W in enumerate(self.weights):
            num_grad = np.zeros_like(W)
            for idx in np.ndindex(W.shape):
                original_value = W[idx]

                # Compute f(W + epsilon)
                W[idx] = original_value + epsilon
                plus_loss = 0.5 * np.mean((self.forward(x) - y) ** 2)

                # Compute f(W - epsilon)
                W[idx] = original_value - epsilon
                minus_loss = 0.5 * np.mean((self.forward(x) - y) ** 2)

                # Restore original value
                W[idx] = original_value

                # Compute numerical gradient
                num_grad[idx] = (plus_loss - minus_loss) / (2 * epsilon)

            numerical_grads.append(num_grad)

        # Restore original weights
        self.weights = original_weights

        # Recompute analytical gradients with restored weights
        self.forward(x)
        analytical_grads = self.backward(x, y)

        # Compare gradients
        for idx, (a_grad, n_grad) in enumerate(zip(analytical_grads, numerical_grads)):
            numerator = np.linalg.norm(a_grad - n_grad)
            denominator = np.linalg.norm(a_grad) + np.linalg.norm(n_grad) + 1e-8
            diff = numerator / denominator
            print(f"Layer {idx + 1} relative difference: {diff}")


if __name__ == "__main__":
    import sys
    import os
    import matplotlib.pyplot as plt

    # just for testing, import activation class and relu for activation function and run a test on synthetic data
    current_dir = os.path.dirname(__file__)
    activations_path = os.path.abspath(os.path.join(current_dir, '../activations'))
    sys.path.append(activations_path)

    from activations_class import *
    from relu import *
    from tanh import *

    # Input and output
    N = 30
    n_features = 2
    out_dim = 1
    epochs = 1000
    X = np.random.randn(n_features, N)
    X.sort()
    y = (2 * (np.arange(N) + np.random.randn(N) * 0.001) / N) - 1
    relu_activation = ActivationFunction(activation=tanh, derivative=tanh_derivative)



    # Define the network
    nn = NeuralNetwork(
        layer_sizes=[n_features, 5, 10, out_dim],
        activation_function=relu_activation,
        last_activation_function=None
    )


    for i in range(epochs):


        # Forward pass
        output = nn.forward(X)
        # Backward pass
        gradients = nn.backward(X, y)
        # Update weights
        nn.update_weights(gradients, learning_rate=0.01)


    for i, t in enumerate(gradients):
        print(f'layer {i + 1} gradients.shape = {t.shape}')

    nn.gradient_check(X, y)

    predicted = nn.forward(X)
    plt.scatter(y, predicted.flatten(), label="Predicted vs Actual", c='r')
    plt.plot(y, y, label="Ideal", linestyle='--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual")
    plt.legend()

    plt.tight_layout()
    plt.show()