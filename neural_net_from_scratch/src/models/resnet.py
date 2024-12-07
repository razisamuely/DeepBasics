import numpy as np
from neural_network import NeuralNetwork

class ResidualNeuralNetwork(NeuralNetwork):
    def __init__(self, layer_sizes, activation_function=None, last_activation_function=None, loss=None):
        super().__init__(layer_sizes, activation_function, last_activation_function, loss)

    def forward(self, X):
        """
        Perform a forward pass through the residual network.
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
            # Store the input for potential residual connection
            residual = X[:-1, :]  # Exclude the bias term

            Z = W @ X
            if i != len(self.weights) - 1:
                if self.activation_function is not None:
                    X = self.activation_function.activation(Z)
                else:
                    X = Z

                # Residual connection: add the original input to the activation output
                if i > 0 and X.shape[0] == residual.shape[0]:
                    X[:, :] += residual

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

        assert self.loss is not None, "Need to define loss function in order to perform backprop"


        gradients = []
        final_output = self.outputs[-1]

        # first calculate the error between prediction and label
        # error = final_output - y
        error = self.loss.loss_gradient(final_output, y)

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

                # Backpropagate through the weight matrix
                new_error = W_no_bias.T @ error

                # Add the identity mapping (residual connection)
                error = new_error + error  # Add the gradient from the residual connection

                # Apply activation derivative
                if self.activation_function:
                    error *= self.activation_function.derivative(self.pre_activations[i - 1])

        # discards the "computation graph" from the forward prop
        self.outputs = []
        self.pre_activations = []

        return gradients


if __name__ == "__main__":
    import sys
    import os
    import matplotlib.pyplot as plt

    # just for testing, import activation class and relu for activation function and run a test on synthetic data
    current_dir = os.path.dirname(__file__)
    activations_path = os.path.abspath(os.path.join(current_dir, '../activations'))
    sys.path.append(activations_path)

    losses_path = os.path.abspath(os.path.join(current_dir, '../losses'))
    sys.path.append(losses_path)

    from activations_class import *
    from relu import *
    from tanh import *


    class LeastSquaresLoss:
        def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
            return np.mean((y_pred - y_true) ** 2)

        def loss_gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
            return (y_pred - y_true) / y_pred.shape[0]


    N_SAMPLES = 32
    n_features = 2
    out_dim = 1
    epochs = 100_000

    def generate_linear_regression_data(bias=3000):
        np.random.seed(42)
        y = np.array([np.array([i + np.random.normal(loc=0, scale=3)]) for i in range(0, N_SAMPLES)]).squeeze() + bias
        X = np.array(np.array([(1, i + np.random.normal(loc=0, scale=3)) for i in range(0, N_SAMPLES)])) / N_SAMPLES

        return X, y


    X, y = generate_linear_regression_data()

    relu_activation = ActivationFunction(activation=relu, derivative=relu_derivative)

    loss = LeastSquaresLoss()

    # Define the network
    nn = ResidualNeuralNetwork(
        layer_sizes=[n_features, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, out_dim],
        activation_function=relu_activation,
        last_activation_function=None,
        loss=loss
    )
    before_weights = nn.weights.copy()

    print(X)

    loss_list = []
    for i in range(epochs):
        # Forward pass
        output = nn.forward(X.T)

        loss_list.append(loss.loss(output, y))

        # Backward pass
        gradients = nn.backward(X.T, y)
        # Update weights
        nn.update_weights(gradients, learning_rate=1e-8)


    # for i, t in enumerate(gradients):
    #     print(f'layer {i + 1} gradients.shape = {t.shape}')
    #     print(f'biases grad = {t[:, -1]}')
    #     print(f'non biases grad = {t[:, :-1]}')

    for i, (W, before_W) in enumerate(zip(nn.weights, before_weights)):
        # The bias terms are in the last column of W
        biases = W[:, -1]
        # print(f'biases_{i + 1} = {biases}')

        # print(f'not biases = {W[:, :-1]}')
        print(biases == before_W[:, -1])


    predicted = nn.forward(X.T)
    plt.scatter(y, predicted.flatten(), label="Predicted vs Actual", c='r')
    plt.plot(y, y, label="Ideal", linestyle='--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual")
    plt.legend()

    # plt.plot(loss_list)

    plt.tight_layout()
    plt.show()