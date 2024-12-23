import numpy as np
import matplotlib.pyplot as plt
import sys
import os
current_dir = os.path.dirname(__file__)
activations_path = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(activations_path)

import pytest
from neural_net_from_scratch.src.optimizers.SGD import SGD
from neural_net_from_scratch.src.losses.sofmax_loss import SoftmaxLoss
from neural_net_from_scratch.src.models.neural_network import DynamicNeuralNetwork


N_SAMPLES = 100
N_FEATURES = 2
N_ITERATIONS = 500
# LEARNING_RATE = 1e-2
# HIDDEN_DIM = 20


def generate_artificial_classification_data_for_softmax_minimazation(n_sampels, n_classes, n_features):
    """Generate synthetic classification data with 3 classes"""
    np.random.seed(42)

    centers = [
        (0, 0),
        (5, 5),
        (-5, 5)
    ]

    X = []
    y = []
    samples_per_class = n_sampels // n_classes

    for class_idx, (cx, cy) in enumerate(centers):
        x_samples = np.random.normal(cx, 1, samples_per_class)
        y_samples = np.random.normal(cy, 1, samples_per_class)
        X.extend([[x, y] for x, y in zip(x_samples, y_samples)])
        y.extend([class_idx] * samples_per_class)

    X = np.array(X)
    y = np.array(y)

    w_init = np.random.randn(n_features + 1, n_classes) * 0.01

    return X, y, w_init


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
    for epoch in range(n_epochs):
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
def test_sgd_softmax_optimization_on_artifical_data(N_SAMPLES, LEARNING_RATE, HIDDEN_DIM, test_size=0.3):
    N_CLASSES = 3
    X, y, w_init = generate_artificial_classification_data_for_softmax_minimazation(N_SAMPLES, N_CLASSES, N_FEATURES)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    train_indices = indices[:int(X.shape[0] * test_size)]
    test_indices = indices[int(X.shape[0] * test_size):]
    X_test, X_train = X[test_indices], X[train_indices]
    y_test, y_train = y[test_indices], y[train_indices]

    train_mean, train_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - train_mean) / train_std

    # use same mean and std for honest checking
    X_test = (X_test - train_mean) / train_std


    one_hot_y_train = one_hot(y_train, N_CLASSES)
    one_hot_y_test = one_hot(y_test, N_CLASSES)
    print(f'X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}')
    print(f'X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}')

    print(f'running full network test with iterations = {N_ITERATIONS}, LR = {LEARNING_RATE}, examples = {N_SAMPLES}')


    layer_configs = [
        {"type": "layer", "input_dim": N_FEATURES, "output_dim": HIDDEN_DIM, "activation": "relu"},
        {"type": "layer", "input_dim": HIDDEN_DIM, "output_dim": HIDDEN_DIM, "activation": "relu"},
        {"type": "layer", "input_dim": HIDDEN_DIM, "output_dim": N_CLASSES, "activation": "tanh"},
    ]

    # Create the network
    network = DynamicNeuralNetwork(layer_configs)

    losses, train_accuracies, test_accuracies = train_network(
        network=network,
        X_train=X_train.T,
        y_train=one_hot_y_train.T,
        X_test=X_test.T,
        y_test=one_hot_y_test.T,
        n_epochs=N_ITERATIONS,
        learning_rate=LEARNING_RATE
    )

    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(train_accuracies)
    ax[0].plot(test_accuracies)
    ax[0].legend(['Train', 'Test'])
    ax[0].set_title(f'Learning rate={LEARNING_RATE}, Hidden Dim={HIDDEN_DIM}, N_SAMPLES={N_SAMPLES}\nAccuracies vs Epochs')
    ax[1].plot(losses)
    ax[1].set_title('Loss vs Epochs')
    plt.tight_layout()
    plt.show()

    title = f'Softmax Classification using a full neural network on test data\nLearning Rate: {LEARNING_RATE}, Hidden Dim: {HIDDEN_DIM}, Num Examples: {N_SAMPLES}'

    plot_softmax_decision_boundary(X_test, y_test.T, network, n_classes=N_CLASSES, title=title, N_SAMPLES=N_SAMPLES)

    # plt.show()


if __name__ == "__main__":

    # # test with 1000 examples
    # N_SAMPLES = 1000
    #
    # # too high learning rate
    # test_sgd_softmax_optimization_on_artifical_data(N_SAMPLES, LEARNING_RATE=1e-0, HIDDEN_DIM=20)
    #
    # # low hidden dim
    # test_sgd_softmax_optimization_on_artifical_data(N_SAMPLES, LEARNING_RATE=1e-2, HIDDEN_DIM=2)
    #
    # # optimal params
    # test_sgd_softmax_optimization_on_artifical_data(N_SAMPLES, LEARNING_RATE=1e-2, HIDDEN_DIM=20)
    #
    #
    # # test with 200 examples
    # N_SAMPLES = 200
    # test_sgd_softmax_optimization_on_artifical_data(N_SAMPLES)

    test_cases = [
        (1000, 1e-0, 20),  # Too high learning rate
        (1000, 1e-2, 2),  # Low hidden dimension
        (1000, 1e-2, 20),  # Optimal parameters
        (200, 1e-2, 20),  # Test with 200 examples
    ]

    for N_SAMPLES, LEARNING_RATE, HIDDEN_DIM in test_cases:
        print(f"Manually testing with N_SAMPLES={N_SAMPLES}, LEARNING_RATE={LEARNING_RATE}, HIDDEN_DIM={HIDDEN_DIM}")
        # Directly call the test function
        test_sgd_softmax_optimization_on_artifical_data(N_SAMPLES, LEARNING_RATE, HIDDEN_DIM)
