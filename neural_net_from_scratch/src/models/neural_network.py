import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu'):
        """
        Initialize a standard neural network.
        
        Args:
            layer_sizes (list): List of integers representing the size of each layer
            activation (str): Activation function to use.
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights = []
        self.biases = []
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights and biases for all layers."""
        for i in range(len(self.layer_sizes) - 1):
            self.weights.append(
                np.random.randn(self.layer_sizes[i+1], self.layer_sizes[i]) * np.sqrt(2/self.layer_sizes[i])
            )
            self.biases.append(np.zeros((self.layer_sizes[i+1], 1)))
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (np.ndarray): Input data
            
        Returns:
            list: List of activations for each layer
        """
        # TODO: Implement forward pass
        pass
    
    def backward(self, x, y, activations):
        """
        Backward pass through the network.
        
        Args:
            x (np.ndarray): Input data
            y (np.ndarray): True labels
            activations (list): List of activations from forward pass
            
        Returns:
            tuple: Gradients for weights and biases
        """
        # TODO: Implement backward pass
        pass

