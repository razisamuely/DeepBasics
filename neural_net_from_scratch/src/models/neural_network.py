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
        self.grad = True

    def _initialize_parameters(self):
        """Initialize weights for all layers."""
        for i in range(len(self.layer_sizes) - 1):
            # +1 for augmented bias input in the current layer
            self.weights.append(
                np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i] + 1) * np.sqrt(2 / (self.layer_sizes[i]))
            )

    def no_grad(self, bool):
        self.grad = bool
        print(f'The neural network will {"NOT" if not self.grad else ""} follow the gradients')


    def forward(self, X):
        """
        Perform a forward pass through the network.
        Keeping track of the gradients only if self.grad is True

        Args:
            X (ndarray): Input data of shape (n_features, n_samples).

        Returns:
            ndarray: Output of the network.
        """

        self.outputs = []  # clear previous outputs
        self.pre_activations = []  # to store pre-activation values
        X = np.vstack([X, np.ones(X.shape[1])])  # augment input with bias instead of keeping a "b" variable

        if self.grad:
            self.outputs.append(X)

        for i, W in enumerate(self.weights):
            Z = W @ X
            if i != len(self.weights) - 1:
                if self.activation_function is not None:
                    X = self.activation_function.activation(Z)
                else:
                    X = Z
                X = np.vstack([X, np.ones(X.shape[1])])  # add the bias for the next layer
            else:
                X = Z

            if self.grad:
                self.pre_activations.append(Z)
                self.outputs.append(X)

        if self.last_activation_function is not None:
            X = self.last_activation_function.activation(X)

            if self.grad:
                self.outputs[-1] = X  # replace the last output with the final activated output
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
        assert self.grad and len(self.outputs) != 0 and len(self.pre_activations) != 0,\
            "Need to define grad=True and run forward prop before performing backprop"


        gradients = []
        final_output = self.outputs[-1]

        # first calculate the error between prediction and label
        error = final_output - y

        # calculate the derivative of the last activation function and multiply it by the error (first delta term)
        if self.last_activation_function:
            error *= self.last_activation_function.derivative(self.pre_activations[-1])

        # backpropagation through each layer
        for i in reversed(range(len(self.weights))):
            # get output of the ith layer
            output = self.outputs[i]

            # calculate dw using delta * output (normalize by batch size)
            grad_w = error @ output.T / x.shape[1]
            gradients.insert(0, grad_w)

            if i > 0:
                W_no_bias = self.weights[i][:, :-1]  # exclude the bias term

                # calculate delta again and repeat the process
                error = W_no_bias.T @ error
                if self.activation_function:
                    error *= self.activation_function.derivative(self.pre_activations[i - 1])

        # discards the "computation graph" from the forward prop
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