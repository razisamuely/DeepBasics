import unittest
import pytest
import numpy as np
from src.optimizers.sgd import SGD, SGDMomentum


class TestOptimizer(unittest.TestCase):
    def test_least_squares(self):
        """Test optimizer on least squares problem."""
        # TODO: Implement optimizer tests
        pass
    
    def test_softmax_optimization(self):
        """Test optimizer on softmax classification."""
        # TODO: Implement softmax optimization tests
        pass

    def test_sgd_step():
        """Test SGD update step with known gradients"""

        optimizer = SGD(learning_rate=0.1)
        params = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        gradients = [np.array([[0.1, 0.2], [0.3, 0.4]])]
        expected = [np.array([[0.99, 1.98], [2.97, 3.96]])]
        
        updated_params = optimizer.step(params, gradients)
        
        assert np.allclose(updated_params[0], expected[0])

    def test_sgd_momentum_step():
        """Test SGD with momentum update step"""
        optimizer = SGDMomentum(learning_rate=0.1, momentum=0.9)
        params = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        gradients = [np.array([[0.1, 0.2], [0.3, 0.4]])]
        
        # Two steps to test momentum
        updated_params1 = optimizer.step(params, gradients)
        updated_params2 = optimizer.step(updated_params1, gradients)
        
        assert updated_params2[0].shape == params[0].shape
        # Momentum should make second update larger than first
        assert np.all(np.abs(updated_params2[0] - updated_params1[0]) > 
                    np.abs(updated_params1[0] - params[0]))