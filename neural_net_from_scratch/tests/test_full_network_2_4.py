import numpy as np
import matplotlib.pyplot as plt


N_SAMPLES = 100
N_FEATURES = 2
N_ITERATIONS = 30_000
LEARNING_RATE = 1e-2
HIDDEN_DIM = N_FEATURES * 10



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
                  X, y,
                  learning_rate=1e-3,
                  n_epochs=20000):
    losses = []
    accuracies = []
    optimizer = SGD(model=network,
                    learning_rate=learning_rate,
                    )

    softmax = SoftmaxLoss()
    for epoch in range(n_epochs):
        out = network.forward(X)

        loss = softmax.loss(out, y)

        grad = softmax.loss_gradient(out, y)

        network.backward(grad)

        optimizer.step()

        losses.append(loss)

        predictions = np.argmax(out, axis=0)
        y_classes = np.argmax(y, axis=0)
        accuracy = (predictions == y_classes).sum()

        accuracies.append(accuracy)

    return losses, accuracies


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
    if N_SAMPLES == 200:
        plt.savefig('../../neural_net_from_scratch/artifacts/full_network_decision_boundary_2_5.png')
    else:
        plt.savefig('../../neural_net_from_scratch/artifacts/full_network_decision_boundary_2_4.png')
    plt.close()
    # plt.show()


def one_hot( y:np.ndarray, n_classes:int) -> np.ndarray:
    n_samples = y.shape[0]
    y_one_hot = np.zeros((n_samples, n_classes))
    y_one_hot[np.arange(n_samples), y] = 1
    return y_one_hot

def test_sgd_softmax_optimization_on_artifical_data(N_SAMPLES):
    N_CLASSES = 3
    X, y, w_init = generate_artificial_classification_data_for_softmax_minimazation(N_SAMPLES, N_CLASSES, N_FEATURES)

    X = (X - X.mean(axis=0)) / X.std(axis=0)



    one_hot_y = one_hot(y, N_CLASSES)
    print(f'x.shape = {X.shape}, y.shape = {y.shape}')

    print(f'running full network test with iterations = {N_ITERATIONS}, LR = {LEARNING_RATE}, examples = {N_SAMPLES}')


    layer_configs = [
        {"type": "layer", "input_dim": N_FEATURES, "output_dim": HIDDEN_DIM, "activation": "relu"},
        {"type": "layer", "input_dim": HIDDEN_DIM, "output_dim": HIDDEN_DIM, "activation": "relu"},
        {"type": "layer", "input_dim": HIDDEN_DIM, "output_dim": N_CLASSES, "activation": "tanh"},
    ]

    # Create the network
    network = DynamicNeuralNetwork(layer_configs)

    losses, accuracies = train_network(
        network=network,
        X=X.T,
        y=one_hot_y.T,
        n_epochs=N_ITERATIONS,
        learning_rate=LEARNING_RATE
    )

    title = f'Softmax Classification using a full neural network\nLearning Rate: {LEARNING_RATE}, Iterations: {N_ITERATIONS}, Num Examples: {N_SAMPLES}'

    plot_softmax_decision_boundary(X, y.T, network, n_classes=N_CLASSES, title=title, N_SAMPLES=N_SAMPLES)


if __name__ == "__main__":
    print("Testing full network optimization...")
    import sys
    import os

    # just for testing, import activation class and relu for activation function and run a test on synthetic data
    current_dir = os.path.dirname(__file__)
    activations_path = os.path.abspath(os.path.join(current_dir, '../../'))
    sys.path.append(activations_path)

    from neural_net_from_scratch.src.optimizers.SGD import SGD
    from neural_net_from_scratch.src.losses.sofmax_loss import SoftmaxLoss
    from neural_net_from_scratch.src.models.neural_network_raz import DynamicNeuralNetwork
    from utils import generate_artificial_classification_data_for_softmax_minimazation


    # test with 100 examples
    N_SAMPLES = 100
    test_sgd_softmax_optimization_on_artifical_data(N_SAMPLES)

    # test with 200 examples
    N_SAMPLES = 200
    test_sgd_softmax_optimization_on_artifical_data(N_SAMPLES)
