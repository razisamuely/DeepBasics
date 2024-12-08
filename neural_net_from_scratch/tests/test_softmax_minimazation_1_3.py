import numpy as np
from neural_net_from_scratch.src.optimizers.SGD import SGD
from neural_net_from_scratch.src.losses.sofmax_loss import SoftmaxLoss
import matplotlib.pyplot as plt
from neural_net_from_scratch.tests.utils import (plot_softmax_decision_boundary, 
                   generate_artificial_classification_data_for_softmax_minimazation, 
                   calculate_accuracy,
                   plot_loss_vs_accuracy_train_vs_test)

from scipy.io import loadmat

N_SAMPLES = 100
N_CLASSES = 3
N_FEATURES = 2
BATCH_SIZE = 32
N_ITERATIONS = 1500
LEARNING_RATE = 1e-3


def test_sgd_softmax_optimization_on_artifical_data():
    X, y, w_init = generate_artificial_classification_data_for_softmax_minimazation(N_SAMPLES, N_CLASSES, N_FEATURES)

    ratio  = 0.8
    split_idx = int(ratio * X.shape[0])
    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    
    model = SoftmaxLoss()
    optimizer = SGD(
        model=model,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        number_of_iteration=N_ITERATIONS
    )

    w_optimized, losses, metrics_tracker, val_metrics_tracker = optimizer.run(X_train, y_train, w_init, metrics=calculate_accuracy, X_val=X_test, y_val=y_test)
    
    plot_softmax_decision_boundary(X, y, model, w_optimized, n_classes=N_CLASSES, title=
                         f'Softmax Classification with SGD\nLearning Rate: {LEARNING_RATE}, Iterations: {N_ITERATIONS}')
    
    
    plot_loss_vs_accuracy_train_vs_test(losses, metrics_tracker, val_metrics_tracker, title='Softmax Classification with SGD\nLoss and Accuracy',
                          path= 'neural_net_from_scratch/artifacts/softmax_sgd_loss_accuracy_train_test_1_3.png')



def test_sgd_softmax_optimization_on_swiss_roll_data():

    data = loadmat('neural_net_from_scratch/data/SwissRollData.mat')
    X_train = data['Yt'].T  
    y_train = np.argmax(data['Ct'], axis=0)  

    X_val = data['Yv'].T 
    y_val = np.argmax(data['Cv'], axis=0)

    unique_labels = set(y_train) | set(y_val)

    N_CLASSES = len(unique_labels)

    N_FEATURES = X_train.shape[1]
    BATCH_SIZE = 32
    N_ITERATIONS = 30
    LEARNING_RATE = 1e-3

    w_init = np.random.randn(N_FEATURES + 1, N_CLASSES) * 0.01


    model = SoftmaxLoss()
    optimizer = SGD(
        model=model,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        number_of_iteration=N_ITERATIONS
    )

    w_optimized, losses, metrics_tracker, val_metrics_tracker = optimizer.run(X_train, y_train, w_init, metrics=calculate_accuracy, X_val=X_val, y_val=y_val)


    plot_softmax_decision_boundary(X_train, y_train, model, w_optimized, n_classes=N_CLASSES, title= "Softmax Classification with SGD on Swiss Roll Data\nLearning Rate: {LEARNING_RATE}, Iterations: {N_ITERATIONS}")

    plot_loss_vs_accuracy_train_vs_test(losses, metrics_tracker, val_metrics_tracker, title='Softmax Classification with SGD on Swiss Roll Data\nLoss and Accuracy',
                            path= 'neural_net_from_scratch/artifacts/softmax_sgd_loss_accuracy_train_test_swiss_roll.png')



if __name__ == "__main__":
    test_sgd_softmax_optimization_on_artifical_data()