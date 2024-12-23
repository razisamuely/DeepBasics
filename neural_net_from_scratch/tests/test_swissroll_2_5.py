import numpy as np
import matplotlib.pyplot as plt
import sys
import os
current_dir = os.path.dirname(__file__)
activations_path = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(activations_path)

import tqdm
import pytest
from neural_net_from_scratch.src.optimizers.SGD import SGD
from neural_net_from_scratch.src.losses.sofmax_loss import SoftmaxLoss
from neural_net_from_scratch.src.models.neural_network import DynamicNeuralNetwork
import scipy
N_SAMPLES = 100
N_FEATURES = 2
N_ITERATIONS = 500
LEARNING_RATE = 1e-2
HIDDEN_DIM = 20
N_CLASSES = 2




def plot_analytical_solution_vs_sgd(X, y, w_sgd, w_analytical, analytical_weights, sgd_weights):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 1:], y, s=10)
    plt.plot(X[:, 1:], X @ w_sgd, color='red', linewidth=2, label='SGD')
    plt.plot(X[:, 1:], X @ w_analytical, color='blue', linewidth=2, label='Analytical')
    plt.grid(True)
    plt.title(f"""SGD  vs Analytical\nSGD solution: {sgd_weights}\nanalytical solution: {analytical_weights}""")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.savefig('neural_net_from_scratch/artifacts/sgd_vs_analytical.png')


def train_network(network,
                  X_train, y_train,
                  X_test, y_test,
                  learning_rate=1e-3,
                  n_epochs=20000):
    losses = []
    train_accuracies = []
    test_accuracies = []
    optimizer = SGD(model=network,
                    learning_rate=learning_rate,
                    )

    softmax = SoftmaxLoss()
    for epoch in tqdm.tqdm(range(n_epochs)):
        train_out = network.forward(X_train)

        loss = softmax.loss(train_out, y_train)

        grad = softmax.loss_gradient(train_out, y_train)

        network.backward(grad)

        optimizer.step()

        losses.append(loss)

        # train metrics
        train_predictions = np.argmax(train_out, axis=0)
        y_classes = np.argmax(y_train, axis=0)
        train_accuracy = (train_predictions == y_classes).sum() / len(train_predictions)

        # test metrics
        test_out = network.forward(X_test)
        test_predictions = np.argmax(test_out, axis=0)
        y_classes = np.argmax(y_test, axis=0)
        test_accuracy = (test_predictions == y_classes).sum() / len(test_predictions)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(f"Epoch: {epoch}, Loss: {np.mean(losses[-100:]):.2}, Train Accuracy: {np.mean(train_accuracies[-100:]):.2}, Test Accuracy: {np.mean(test_accuracies[-100:]):.2}")

    return losses, train_accuracies, test_accuracies


def plot_softmax_decision_boundary(X, y, model, n_classes, title, N_SAMPLES):
    """Plot the decision boundary and data points"""
    plt.figure(figsize=(10, 8))

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(mesh_points.T)
    Z = np.argmax(Z, axis=0)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)

    colors = ['red', 'blue', 'green']
    for i in range(n_classes):
        idx = y == i
        plt.scatter(X[idx, 0], X[idx, 1], c=colors[i], label=f'Class {i}')

    plt.title(title)
    plt.legend()
    plt.grid(True)
    # if N_SAMPLES == 200:
    #     plt.savefig('../../neural_net_from_scratch/artifacts/full_network_decision_boundary_2_5.png')
    # else:
    #     plt.savefig('../../neural_net_from_scratch/artifacts/full_network_decision_boundary_2_4.png')
    # plt.close()
    plt.show()


def one_hot( y:np.ndarray, n_classes:int) -> np.ndarray:
    n_samples = y.shape[0]
    y_one_hot = np.zeros((n_samples, n_classes))
    y_one_hot[np.arange(n_samples), y] = 1
    return y_one_hot


@pytest.mark.parametrize("N_SAMPLES, LEARNING_RATE, HIDDEN_DIM", [
    (1000, 1e-0, 20),   # Too high learning rate
    (1000, 1e-2, 2),    # Low hidden dimension
    (1000, 1e-2, 20),   # Optimal parameters
    (200, 1e-2, 20),    # Test with 200 examples
])

def main():
    data = scipy.io.loadmat('neural_net_from_scratch/data/SwissRollData.mat')
    
    X_train = data['Ct'].T
    y_train = data['Yt'].T

    
    X_val = data['Cv'].T
    y_val = data['Yv'].T
    
    
    train_mean, train_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std


    layer_configs = [
        {"type": "layer", "input_dim": N_FEATURES, "output_dim": HIDDEN_DIM, "activation": "relu"},
        {"type": "layer", "input_dim": HIDDEN_DIM, "output_dim": HIDDEN_DIM, "activation": "relu"},
        {"type": "layer", "input_dim": HIDDEN_DIM, "output_dim": N_CLASSES, "activation": "tanh"},
    ]

    # # Create the network
    network = DynamicNeuralNetwork(layer_configs)

    losses, train_accuracies, test_accuracies = train_network(
        network=network,
        X_train=X_train.T,
        y_train=y_train.T,
        X_test=X_val.T,
        y_test=y_val.T,
        n_epochs=N_ITERATIONS,
        learning_rate=LEARNING_RATE
    )

    # fig, ax = plt.subplots(nrows=2, ncols=1)
    # ax[0].plot(train_accuracies)
    # ax[0].plot(test_accuracies)
    # ax[0].legend(['Train', 'Test'])
    # ax[0].set_title(f'Learning rate={LEARNING_RATE}, Hidden Dim={HIDDEN_DIM}, N_SAMPLES={N_SAMPLES}\nAccuracies vs Epochs')
    # ax[1].plot(losses)
    # ax[1].set_title('Loss vs Epochs')
    # plt.tight_layout()
    # plt.show()

    # title = f'Softmax Classification using a full neural network on test data\nLearning Rate: {LEARNING_RATE}, Hidden Dim: {HIDDEN_DIM}, Num Examples: {N_SAMPLES}'

    # plot_softmax_decision_boundary(X_test, y_test.T, network, n_classes=N_CLASSES, title=title, N_SAMPLES=N_SAMPLES)



if __name__ == "__main__":

    main()
