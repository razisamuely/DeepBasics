import numpy as np
from typing import List, Tuple, Callable, Union

class Activation:
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x)**2
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        s = Activation.sigmoid(x)
        return s * (1 - s)

class Layer:
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 activation: str = 'relu' ):
        self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2.0/input_dim)  # He initialization
        self.b = np.zeros((output_dim, 1))
        
        if activation == 'relu':
            self.activation = Activation.relu
            self.activation_derivative = Activation.relu_derivative
        elif activation == 'tanh':
            self.activation = Activation.tanh
            self.activation_derivative = Activation.tanh_derivative
        elif activation == 'sigmoid':
            self.activation = Activation.sigmoid
            self.activation_derivative = Activation.sigmoid_derivative
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Initialize gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        # Cache for backward pass
        self.x = None
        self.z = None
        self.a = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z = np.dot(self.W, x) + self.b
        self.a = self.activation(self.z)
        
        return self.a

    def backward(self, dL_da: np.ndarray) -> np.ndarray:
        da_dz = self.activation_derivative(self.z)
        
        # (45) σ'(Wx + b) ⊙ v
        dL_dz = dL_da * da_dz
        
        # (46): (σ'(WX + b) ⊙ V)X^T | Relative to W
        self.dW = np.dot(dL_dz, self.x.T)
        
        # (45): sum over columns of σ'(WX + b) ⊙ V | Relative to b
        self.db = np.sum(dL_dz, axis=1, keepdims=True)
        
        # (P21): W^T(σ'(Wx + b) ⊙ v) | Relative to x
        dL_dx = np.dot(self.W.T, dL_dz)
        
        return dL_dx

class DynamicNeuralNetwork:
    def __init__(self, layer_dims: List[int], activations: List[str]):
        """
        Initialize a dynamic neural network
        
        Args:
            layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
            activations: List of activation functions for each layer
        """
        self.layers = []
        for i in range(len(layer_dims) - 1):
            layer = Layer(
                input_dim=layer_dims[i],
                output_dim=layer_dims[i + 1],
                activation=activations[i]
            )
            self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        current_activation = x
        for layer in self.layers:
            current_activation = layer.forward(current_activation)
        return current_activation

    def backward(self, dL_dy: np.ndarray) -> None:
        current_gradient = dL_dy
        for layer in reversed(self.layers):
            current_gradient = layer.backward(current_gradient)

    def update_parameters(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.W -= learning_rate * layer.dW
            layer.b -= learning_rate * layer.db

# Example usage
def test_dynamic_network():
    # Define network architecture
    layer_dims = [784, 256, 128, 64, 10]  # Example for MNIST
    activations = ['relu', 'relu', 'relu', 'sigmoid']
    residual_connections = [False, False, False, False]  # Example with one residual connection
    
    # Create network
    network = DynamicNeuralNetwork(
        layer_dims=layer_dims,
        activations=activations,
        residual_connections=residual_connections
    )
    
    # Test forward pass
    x = np.random.randn(784, 32)  # 32 samples
    output = network.forward(x)
    
    # Test backward pass
    dL_dy = np.random.randn(*output.shape)
    network.backward(dL_dy)
    
    # Update parameters
    network.update_parameters(learning_rate=0.01)
    
    return output

if __name__ == "__main__":
    output = test_dynamic_network()
    print(f"Output shape: {output.shape}")