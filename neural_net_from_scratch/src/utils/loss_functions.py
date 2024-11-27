import numpy as np

class SoftmaxLoss:
    @staticmethod
    def forward(logits, labels):
        """
        Compute softmax loss.
        
        Args:
            logits (np.ndarray): Raw output from the network
            labels (np.ndarray): True labels
            
        Returns:
            float: Loss value
        """
        # TODO: Implement softmax loss
        pass
    
    @staticmethod
    def backward(logits, labels):
        """
        Compute gradients of softmax loss.
        
        Args:
            logits (np.ndarray): Raw output from the network
            labels (np.ndarray): True labels
            
        Returns:
            np.ndarray: Gradients with respect to logits
        """
        # TODO: Implement softmax loss gradient
        pass