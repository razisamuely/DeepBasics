import numpy as np
from typing import List, Tuple, Callable, Union, Dict
import matplotlib.pyplot as plt


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
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init_weights(input_dim, output_dim)
        self.init_gradients()
        
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
        

        self.x = None
        self.z = None
        self.a = None

    def init_weights(self, input_dim: int, output_dim: int) -> None:
        self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2.0/input_dim)
        self.b = np.zeros((output_dim, 1))
    
    def init_gradients(self) -> None:
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

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
    
    def update_parameters(self, learning_rate: float) -> None:
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db


class ResidualBlock:
    def __init__(self, input_dim: int, output_dim: int, activation: str):
        self.layers = [
            Layer(input_dim, output_dim, activation),
            Layer(output_dim, output_dim, activation)
        ]
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        a = self.layers[0].forward(x)
        a = self.layers[1].forward(a)
        return a + x
    
    def backward(self, dL_da: np.ndarray) -> np.ndarray:
        dL_dx_second = self.layers[1].backward(dL_da)
        dL_dx_first = self.layers[0].backward(dL_dx_second)
        dL_dx =  dL_da + dL_dx_first
        return dL_dx

    def update_parameters(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.update_parameters(learning_rate)


class DynamicNeuralNetwork:
    def __init__(self, layer_configs: List[Dict]):
        self.layers = []
        for config in layer_configs:
            if config["type"] == "layer":
                layer = Layer(
                    input_dim=config["input_dim"],
                    output_dim=config["output_dim"],
                    activation=config["activation"]
                )
                self.layers.append(layer)
            elif config["type"] == "residual":
                residual_block = ResidualBlock(
                    input_dim=config["input_dim"],
                    output_dim=config["output_dim"],
                    activation=config["activation"]
                )
                self.layers.append(residual_block)
                
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
            layer.update_parameters(learning_rate)



if __name__ == "__main__":
    final_output_dim = 8
    layer_configs = [
        {"type": "layer", "input_dim": 4, "output_dim": 8, "activation": "relu"},
        {"type": "residual", "input_dim": 8, "output_dim": final_output_dim, "activation": "relu"},
    ]
    
    # Create the network
    network = DynamicNeuralNetwork(layer_configs)

    # Input data
    x = np.random.randn(4, 10)  # 4 features, 10 samples

    # Forward pass
    output = network.forward(x)

    # Compute gradient with matching dimensions
    dL_dy = np.random.randn(final_output_dim, 10)  # Changed from (2,10) to (8,10) to match output

    # Backward pass
    network.backward(dL_dy)

    # Update parameters
    network.update_parameters(learning_rate=0.01)

    # print forward pass output
    print(output)