class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate (float): Learning rate for optimization
            momentum (float): Momentum factor
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def step(self, params, gradients):
        """
        Perform one optimization step.
        
        Args:
            params (list): List of parameters to update
            gradients (list): List of gradients for each parameter
            
        Returns:
            list: Updated parameters
        """
        # TODO: Implement SGD update step
        pass

class SGDMomentum(SGD):
    """SGD with momentum implementation."""
    def step(self, params, gradients):
        """Implement momentum update step."""
        # TODO: Implement momentum update
        pass