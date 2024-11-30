import numpy as np
from neural_net_from_scratch.src.utils.utils import augment_features
from neural_net_from_scratch.src.utils.utils import one_hot

class LeastSquaresLoss:
    def mean_squared_error(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean((y_pred - y_true) ** 2)

    def mse_gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return (y_pred - y_true) / y_pred.shape[0]

    def w_gradient(self, X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return  X.T @ self.mse_gradient(y_pred, y_true)

    def forward(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        return X @ W

    def gradients(self, X: np.ndarray, y: np.ndarray , W: np.ndarray) -> np.ndarray:
        y_pred = self.forward(X, W)
        
        if (len(W.shape) > 1) and (W.shape[1] > 1):
            y_true = one_hot(y, W.shape[1])
        else:
            y_true = y

            
        gradient = self.w_gradient(X, y_pred, y_true)
        return gradient


N_SAMPLES = 10000
N_FEATURES = 2
BATCH_SIZE = 100
N_ITERATIONS = 200
LEARNING_RATE = 1e-8
NOISE_SCALE = 3

def generate_linear_regression_data():
    np.random.seed(42)
    # Generate X without intercept column (it will be added in augment_features)
    X = np.array([(i + np.random.normal(0, 1),) for i in range(N_SAMPLES)])
    y = np.array([i + np.random.normal(0, NOISE_SCALE) for i in range(N_SAMPLES)])
    # Initialize weights for augmented features (including bias)
    w_init = np.zeros(N_FEATURES)  # N_FEATURES already includes bias term
    return X, y, w_init

if __name__ == "__main__":
    n_samples, n_features = 100, 4
    n_classes = 3

    X, y, w_init = generate_linear_regression_data()
    X = augment_features(X)
    model = LeastSquaresLoss()
    
    # Test forward
    y_pred = model.forward(X, w_init)

    # Test mean squared error
    loss = model.mean_squared_error(y_pred, y)  
    print("\nTest mean squared error ", loss)

    # Test mse gradient
    gradient = model.mse_gradient(y_pred, y)
    print("\nTest mse gradient ", gradient)

    # Test w gradient
    w_grad = model.w_gradient(X, y_pred, y)
    print("\nTest w gradient ", w_grad)

    # Test gradients
    dW = model.gradients(X, y, w_init)
    print("\nTest gradients ", dW)

   