import pytest
import numpy as np
from src.models.neural_network import NeuralNetwork, ResNet

def test_neural_network_forward():
    """Test forward pass with known input and expected output"""
    input_size = 3
    hidden_size = 4
    output_size = 2
    network = NeuralNetwork([input_size, hidden_size, output_size])
    test_input = np.random.randn(input_size, 1)
    
    output = network.forward(test_input)
    
    assert len(output) == 3  # Input layer, hidden layer, output layer
    assert output[-1].shape == (output_size, 1)

def test_neural_network_backward():
    """Test backward pass gradient shapes"""

    input_size = 3
    hidden_size = 4
    output_size = 2
    network = NeuralNetwork([input_size, hidden_size, output_size])
    test_input = np.random.randn(input_size, 1)
    test_labels = np.array([[1], [0]])
    
    activations = network.forward(test_input)
    gradients = network.backward(test_input, test_labels, activations)
    
    assert len(gradients) == 2  # One for each layer (excluding input)
    assert gradients[0][0].shape == network.weights[0].shape  # Weight gradients
    assert gradients[0][1].shape == network.biases[0].shape   # Bias gradients

def test_resnet_forward():
    """Test ResNet forward pass with known input"""
    input_size = 3
    hidden_size = 3  # Same size for residual connection
    output_size = 2
    network = ResNet([input_size, hidden_size, output_size])
    test_input = np.random.randn(input_size, 1)
    
    output = network.forward(test_input)
    
    assert len(output) == 3
    assert output[-1].shape == (output_size, 1)

def test_resnet_backward():
    """Test ResNet backward pass gradient shapes"""
    input_size = 3
    hidden_size = 3
    output_size = 2
    network = ResNet([input_size, hidden_size, output_size])
    test_input = np.random.randn(input_size, 1)
    test_labels = np.array([[1], [0]])
    
    activations = network.forward(test_input)
    gradients = network.backward(test_input, test_labels, activations)
    
    assert len(gradients) == 2
    assert gradients[0][0].shape == network.weights1[0].shape
    assert gradients[0][1].shape == network.weights2[0].shape