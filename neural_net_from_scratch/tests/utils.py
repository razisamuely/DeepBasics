import numpy as np
from neural_net_from_scratch.src.optimizers.SGD import SGD
from neural_net_from_scratch.src.losses.sofmax_loss import SoftmaxLoss
import matplotlib.pyplot as plt


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

def plot_softmax_decision_boundary(X, y, model, weights,  n_classes, title):
    """Plot the decision boundary and data points"""
    plt.figure(figsize=(10, 8))
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(mesh_points, weights)
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    
    colors = ['red', 'blue', 'green']
    for i in range(n_classes):
        idx = y == i
        plt.scatter(X[idx, 0], X[idx, 1], c=colors[i], label=f'Class {i}')
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.savefig('neural_net_from_scratch/artifacts/softmax_sgd_decision_boundary_1_3.png')
    plt.close()


def plot_loss_accuracy(losses, accuracy,title, path):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(accuracy, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(title)
    plt.grid(True)
    plt.savefig(path)

def calculate_accuracy(model, X, y, w):
        y_pred = np.argmax(model.forward(X, w), axis = 1)
        return np.mean(y_pred == y)


def plot_loss_vs_accuracy_train_vs_test(losses, train_accuracy, test_accuracy, title, path):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color_train_loss = 'tab:red'
        color_test_loss = 'tab:orange'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        train_loss_line = ax1.plot(losses, color=color_train_loss, label='Train Loss')
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()
        color_train_acc = 'tab:blue'
        color_test_acc = 'tab:green'
        ax2.set_ylabel('Accuracy')
        train_acc_line = ax2.plot(train_accuracy, color=color_train_acc, label='Train Accuracy')
        test_acc_line = ax2.plot(test_accuracy, color=color_test_acc, label='Test Accuracy')
        ax2.tick_params(axis='y')

        lines = train_loss_line + train_acc_line + test_acc_line
        labels = [line.get_label() for line in lines]
        
        plt.legend(lines, labels, loc='center left', bbox_to_anchor=(1.15, 0.5))

        plt.title(title)
        plt.grid(True)
        
        plt.tight_layout()
        
        plt.savefig(path, bbox_inches='tight')
        plt.close()
