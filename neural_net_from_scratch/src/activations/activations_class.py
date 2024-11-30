import numpy as np


class ActivationFunction:
    """
    A class representing an activation function and its derivative.
    """

    def __init__(self, activation, derivative):
        """
        Initialize the activation function.

        Parameters:
        - activation: A callable that computes the activation function.
        - derivative: A callable that computes the derivative of the activation function.
        """
        self.activation = activation
        self.derivative = derivative




if __name__ == "__main__":
    from relu import *
    from tanh import *

    # Example usage
    tanh_activation = ActivationFunction(activation=tanh, derivative=tanh_derivative)
    relu_activation = ActivationFunction(activation=relu, derivative=relu_derivative)

    # Test with example input
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # Using tanh activation
    print("Using tanh:")
    print("Activation:", tanh_activation.activation(x))
    print("Derivative:", tanh_activation.derivative(x))

    # Using ReLU activation
    print("\nUsing ReLU:")
    print("Activation:", relu_activation.activation(x))
    print("Derivative:", relu_activation.derivative(x))
