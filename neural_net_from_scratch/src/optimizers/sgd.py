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
    
    def step(self, params_n, gradients):
        params_n_1 = params_n - self.learning_rate * gradients
        return params_n_1
    
    def get_start_end_batch_index(self, batch_index):
        start = batch_index * self.batch_size
        end = start + self.batch_size
        return start, end

    def get_batche(self, X, y , batch_index):
        start, end = self.get_start_end_batch_index(batch_index)
        X_batch = X[start:end]
        y_batch = y[start:end]
        return X_batch, y_batch

    def run(self, X, y, w):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_batches = n_samples // self.batch_size
        losses = []
        for i in tqdm(range(self.number_of_iteration)):
            for j in range(n_batches):
                X_batch, y_batch = self.get_batche(X, y, j)
                gradient = self.model.gradients(X_batch, y_batch, w)
                w = self.step(w, gradient)
                loss = self.model.mean_squared_error(self.model.forward(X, w), y)
                losses.append(loss)

            if i % 100 == 0:
                print(f"Epoch {i} | Loss: {np.mean(losses[:-10])}")
            

        return w

    

        

class SGDMomentum(SGD):
    def step(self, params, gradients):
        pass