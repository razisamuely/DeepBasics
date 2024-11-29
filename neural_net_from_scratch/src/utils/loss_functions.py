import numpy as np

class SoftmaxLoss:
    def sofotmax(self,x:np.ndarray) -> np.ndarray:
        exponent_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs = exponent_x / np.sum(exponent_x, axis=1, keepdims=True)
        return probs
    
    def cross_entropy_loss(self, pred_prob:np.ndarray, labels:np.ndarray, smoothing:float=1e-15) -> float:
        loss = -np.sum(labels * np.log(pred_prob + smoothing)) / len(pred_prob)
        return loss

    def softmax_gradient(self, y_pred:np.ndarray, labels:np.ndarray) -> np.ndarray:
        return y_pred - labels 

    def w_gradient(self, X:np.ndarray, y_pred:np.ndarray, labels:np.ndarray) -> np.ndarray:
        return np.dot(X.T, y_pred - labels)
    
    def augment_features(self, X:np.ndarray) -> np.ndarray:
        return np.hstack([X, np.ones((X.shape[0], 1))])
    
    def forward(self, X:np.ndarray, W:np.ndarray) -> np.ndarray:
        X_aug = self.augment_features(X)
        scores = np.dot(X_aug, W)
        probs = self.sofotmax(scores)
        return probs

    def one_hot(self, y:np.ndarray, n_classes:int) -> np.ndarray:
        n_samples = y.shape[0]
        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y] = 1
        return y_one_hot
    
    def gradients(self, X:np.ndarray, y:np.ndarray, predicted_prob:np.ndarray, W:np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        X_aug = self.augment_features(X)
        self.forward(X, W)
        y_one_hot = self.one_hot(y, W.shape[1])
        dscores = (predicted_prob - y_one_hot) / n_samples
        dW = np.dot(X_aug.T, dscores)
        return dW
    
    
if __name__ == "__main__":
    
    logits = np.array([[1, 2, 3, 2,]]) * 0
    labels = np.array([[0, 1, 0 ,0,]]) * 0

    sfml = SoftmaxLoss()
    
    # Test softmax
    pred_prob = sfml.sofotmax(logits)
    print("\nTest softmax ", pred_prob)

    # Test cross entropy loss
    loss = sfml.cross_entropy_loss(pred_prob, labels)
    print("\nTest cross entropy loss ", loss)

    # Test softmax gradient
    y_pred = sfml.softmax_gradient(pred_prob, labels)
    print("\nTest softmax gradient ", y_pred)

    # Test w gradient
    X = np.array([[1, 2, 3, 2,]])
    w_grad = sfml.w_gradient(X, y_pred, labels)
    print("\nTest w gradient ", w_grad)

    # Test forward
    W = np.random.randn(X.shape[1] + 1, 4)
    pred_prob = sfml.forward(X, W)
    print("\nTest forward ", pred_prob)

    # Test gradients
    dW = sfml.gradients(X, labels, pred_prob, W)
    print("\nTest gradients ", dW)

    # Test one hot
    y = np.array([0, 1, 2, 3])
    n_classes = 4
    one_hot = sfml.one_hot(y, n_classes)
    print("\nTest one hot ", one_hot)


    