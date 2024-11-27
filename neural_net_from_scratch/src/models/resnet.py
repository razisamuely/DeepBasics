class ResNet:
    def __init__(self, layer_sizes, activation='relu'):
        """
        Initialize a ResNet network.
        
        Args:
            layer_sizes (list): List of integers representing the size of each layer
            activation (str): Activation function to use ('relu' or 'sigmoid')
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights1 = []
        self.weights2 = []
        self.biases1 = []
        self.biases2 = []
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights and biases for all layers."""
        # TODO: Implement parameter initialization
        pass
    
    def forward(self, x):
        """Forward pass implementation for ResNet."""
        # TODO: Implement forward pass
        pass
    
    def backward(self, x, y, activations):
        """Backward pass implementation for ResNet."""
        # TODO: Implement backward pass
        pass