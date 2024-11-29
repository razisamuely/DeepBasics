import pytest
import numpy as np
from neural_net_from_scratch.src.utils.loss_functions import SoftmaxLoss

# Constants
N_CLASSES = 4
N_FEATURES = 4
EPSILON = 1e-7

@pytest.fixture
def softmax_loss():
    return SoftmaxLoss()

@pytest.fixture
def simple_data():
    logits = np.array([[1, 2, 3, 2]]) * 0
    labels = np.array([[0, 1, 0, 0]]) * 0
    return logits, labels

@pytest.fixture
def feature_data():
    X = np.array([[1, 2, 3, 2]])
    W = np.random.randn(N_FEATURES + 1, N_CLASSES)
    return X, W

def test_softmax(softmax_loss, simple_data):
    logits, _ = simple_data
    probs = softmax_loss.sofotmax(logits)
    
    # Test probability properties
    assert np.allclose(np.sum(probs, axis=1), 1.0), "Probabilities should sum to 1"
    assert np.all(probs >= 0), "Probabilities should be non-negative"
    assert np.all(probs <= 1), "Probabilities should be <= 1"
    assert probs.shape == logits.shape, "Output shape should match input shape"

def test_cross_entropy_loss(softmax_loss, simple_data):
    logits, labels = simple_data
    probs = softmax_loss.sofotmax(logits)
    loss = softmax_loss.cross_entropy_loss(probs, labels)
    
    assert isinstance(loss, float), "Loss should be a float"
    assert loss >= 0, "Loss should be non-negative"

def test_softmax_gradient(softmax_loss, simple_data):
    logits, labels = simple_data
    probs = softmax_loss.sofotmax(logits)
    gradient = softmax_loss.softmax_gradient(probs, labels)
    
    assert gradient.shape == probs.shape, "Gradient shape should match input shape"

def test_w_gradient(softmax_loss, feature_data, simple_data):
    X, _ = feature_data
    _, labels = simple_data
    probs = np.random.rand(1, N_CLASSES)
    probs = probs / np.sum(probs) 
    
    gradient = softmax_loss.w_gradient(X, probs, labels)
    
    assert gradient.shape == (X.shape[1], N_CLASSES), "Gradient shape should be (n_features, n_classes)"

def test_augment_features(softmax_loss, feature_data):
    X, _ = feature_data
    X_aug = softmax_loss.augment_features(X)
    
    assert X_aug.shape == (X.shape[0], X.shape[1] + 1), "Augmented features should have one more column"
    assert np.all(X_aug[:, -1] == 1), "Last column should be all ones"

def test_forward(softmax_loss, feature_data):
    X, W = feature_data
    probs = softmax_loss.forward(X, W)
    
    assert probs.shape == (X.shape[0], N_CLASSES), "Output shape should be (n_samples, n_classes)"
    assert np.allclose(np.sum(probs, axis=1), 1.0), "Probabilities should sum to 1"
    assert np.all(probs >= 0) and np.all(probs <= 1), "Probabilities should be between 0 and 1"

def test_one_hot(softmax_loss):
    y = np.array([0, 1, 2, 3])
    one_hot = softmax_loss.one_hot(y, N_CLASSES)
    
    assert one_hot.shape == (4, N_CLASSES), "One-hot shape should be (n_samples, n_classes)"
    assert np.all(np.sum(one_hot, axis=1) == 1), "Each row should sum to 1"
    for i, label in enumerate(y):
        assert one_hot[i, label] == 1, f"Wrong one-hot encoding for label {label}"

def test_gradients(softmax_loss, feature_data):
    X, W = feature_data
    y = np.array([0])  
    probs = softmax_loss.forward(X, W)
    
    dW = softmax_loss.gradients(X, y, probs, W)
    
    assert dW.shape == W.shape, "Gradient shape should match weights shape"
    
    eps = 1e-7
    numerical_grad = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_plus = W.copy()
            W_plus[i, j] += eps
            W_minus = W.copy()
            W_minus[i, j] -= eps
            
            probs_plus = softmax_loss.forward(X, W_plus)
            probs_minus = softmax_loss.forward(X, W_minus)
            y_one_hot = softmax_loss.one_hot(y, W.shape[1])
            
            loss_plus = softmax_loss.cross_entropy_loss(probs_plus, y_one_hot)
            loss_minus = softmax_loss.cross_entropy_loss(probs_minus, y_one_hot)
            
            numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)
    
    assert np.allclose(dW, numerical_grad, rtol=1e-4, atol=1e-4), \
        "Analytical and numerical gradients should be close"