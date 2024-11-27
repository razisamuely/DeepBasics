import pytest
import numpy as np
from src.utils.loss_functions import SoftmaxLoss
from src.utils.gradient_checker import check_gradient

def test_softmax_loss_forward():
    """Test softmax loss computation with known values"""
    logits = np.array([[2.0], [1.0], [0.5]])
    labels = np.array([[0], [1], [0]])
    expected_loss = -np.log(np.exp(1.0) / np.sum(np.exp([2.0, 1.0, 0.5])))
    
    loss = SoftmaxLoss.forward(logits, labels)
    
    assert np.isclose(loss, expected_loss)

def test_softmax_loss_backward():
    """Test softmax loss gradients"""
    logits = np.array([[2.0], [1.0], [0.5]])
    labels = np.array([[0], [1], [0]])
    
    gradients = SoftmaxLoss.backward(logits, labels)
    
    assert gradients.shape == logits.shape
    assert np.all(np.abs(gradients) <= 1.0)  # Gradients should be probabilities

def test_gradient_checker():
    """Test gradient checker with simple function"""

    def simple_function(x):
        """x^2 function with known derivative 2x"""
        return x**2, 2*x
    
    test_point = np.array([2.0])
    
    analytical, numerical, error = check_gradient(simple_function, test_point)
    
    assert np.isclose(analytical, numerical, rtol=1e-5)
    assert error < 1e-5

def test_gradient_checker_softmax():
    """Test gradient checker with softmax loss"""
    logits = np.array([[2.0], [1.0], [0.5]])
    labels = np.array([[0], [1], [0]])
    
    def softmax_wrapper(x):
        """Wrapper for softmax loss to match gradient checker interface"""
        loss = SoftmaxLoss.forward(x, labels)
        grad = SoftmaxLoss.backward(x, labels)
        return loss, grad
    
    analytical, numerical, error = check_gradient(softmax_wrapper, logits)
    
    assert error < 1e-5