import numpy as np
import matplotlib.pyplot as plt


class SGD:
    def __init__(self, learning_rate=0.01):
        """
        Initialize SGD optimizer.

        Args:
            learning_rate (float): Learning rate for optimization
        """
        self.learning_rate = learning_rate

    def step(self, params, gradients):
        """
        Perform one optimization step.

        Args:
            params (list of np.ndarray): List of parameter arrays to update
            gradients (list of np.ndarray): List of gradient arrays for each parameter

        Returns:
            list of np.ndarray: Updated parameters
        """
        for i in range(len(params)):
            params[i] -= self.learning_rate * gradients[i]
        return params


class SGDMomentum(SGD):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Initialize SGD with Momentum optimizer.

        Args:
            learning_rate (float): Learning rate for optimization
            momentum (float): Momentum factor
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None  # Will be initialized as an array matching params shape

    def step(self, params, gradients):
        """
        Perform one optimization step with momentum.

        Args:
            params (list of np.ndarray): List of parameter arrays to update
            gradients (list of np.ndarray): List of gradient arrays for each parameter

        Returns:
            list of np.ndarray: Updated parameters
        """
        if self.velocity is None:
            self.velocity = [np.zeros_like(param) for param in params]

        for i in range(len(params)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * gradients[i]
            params[i] += self.velocity[i]

        return params


if __name__ == "__main__":
    def least_squares_loss(params, X, y):
        """
        Compute the least squares loss.

        Args:
            params list (np.ndarray): Parameters [w, b].
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.

        Returns:
            float: Loss value.
        """
        w, b = params
        predictions = X.dot(w) + b
        loss = np.mean((predictions - y) ** 2)
        return loss


    def least_squares_grad(params, X, y):
        """
        Compute the gradient of the least squares loss.

        Args:
            params list (np.ndarray): Parameters [w, b].
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.

        Returns:
            list of np.ndarray: Gradients [dw, db].
        """
        w, b = params
        predictions = X.dot(w) + b
        error = predictions - y
        dw = (2 / len(X)) * X.T.dot(error)
        db = (2 / len(X)) * np.sum(error)
        return [dw, db]


    # Generate synthetic data for testing
    np.random.seed(42)
    N = 100
    X = np.random.rand(N, 1)
    true_w = np.array([2.0])
    true_b = 1.0  # True bias
    y = X.dot(true_w) + true_b + 0.1 * np.random.randn(100)

    epochs = 100

    initial_params_sgd = [np.random.randn(1), np.random.randn()]
    initial_params_momentum = [np.copy(initial_params_sgd[0]), np.copy(initial_params_sgd[1])]

    # initialize optimizers and train model
    sgd_optimizer = SGD(learning_rate=0.1)
    momentum_optimizer = SGDMomentum(learning_rate=0.1, momentum=0.9)

    params_sgd = initial_params_sgd
    loss_history_sgd = []

    for epoch in range(epochs):
        gradients = least_squares_grad(params_sgd, X, y)
        params_sgd = sgd_optimizer.step(params_sgd, gradients)
        loss = least_squares_loss(params_sgd, X, y)
        loss_history_sgd.append(loss)

    params_momentum = initial_params_momentum
    loss_history_momentum = []

    for epoch in range(epochs):
        gradients = least_squares_grad(params_momentum, X, y)
        params_momentum = momentum_optimizer.step(params_momentum, gradients)
        loss = least_squares_loss(params_momentum, X, y)
        loss_history_momentum.append(loss)

    # plotting sanity check, checking if loss decreases
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), loss_history_sgd, label="SGD", linestyle="--")
    plt.plot(range(epochs), loss_history_momentum, label="SGD with Momentum", linestyle="-")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Least Squares Loss Over Epochs for SGD and SGD with Momentum")
    plt.legend()
    plt.grid()
    plt.show()