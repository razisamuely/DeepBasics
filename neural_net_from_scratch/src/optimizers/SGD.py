import numpy as np
from tqdm import tqdm
class SGD:
    def __init__(self,model =None, learning_rate=0.01, momentum=0.9,number_of_iteration =100, batch_size = 4, tolerance = 0.01,sgd_type = 'vanilla'):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
        self.sgd_type = sgd_type
        self.number_of_iteration = number_of_iteration
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.model = model
    
    def step(self, params_n=None, gradients=None):
        if params_n is not None and gradients is not None:
            params_n_1 = params_n - self.learning_rate * gradients
            return params_n_1

        for layer in self.model.layers:
            layer.update_parameters(self.learning_rate)


    def get_start_end_batch_index(self, batch_index):
        start = batch_index * self.batch_size
        end = start + self.batch_size
        return start, end

    def get_batche(self, X, y , batch_index):
        start, end = self.get_start_end_batch_index(batch_index)
        X_batch = X[start:end]
        y_batch = y[start:end]
        return X_batch, y_batch

    def run(self, X, y, w, metrics = None, X_val = None, y_val = None):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_batches = n_samples // self.batch_size
        losses = []
        metrics_tracker = []
        val_metrics_tracker = []
        for i in tqdm(range(self.number_of_iteration)):
            for j in range(n_batches):
                X_batch, y_batch = self.get_batche(X, y, j)
                gradient = self.model.gradients(X_batch, y_batch, w)
                w = self.step(w, gradient)
                loss = self.model.loss(self.model.forward(X, w), y)
                losses.append(loss)
                if metrics:
                    metrics_tracker.append(metrics(self.model, X, y, w))
                if X_val is not None:
                    val_metrics_tracker.append(metrics(self.model, X_val, y_val, w))
            if i % 100 == 0:
                print(f"Epoch {i} | Loss: {np.mean(losses[:-10])}")
            

        if metrics:
            return w, losses, metrics_tracker, val_metrics_tracker
        
        else:
            return w, losses
    
    def calculate_accuracy(model, X, y, w):
        y_pred = np.argmax(model.forward(X, w), axis = 1)
        return np.mean(y_pred == y)
        

class SGDMomentum(SGD):
    def step(self, params, gradients):
        pass