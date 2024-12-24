import numpy as np
from neural_net_from_scratch.src.optimizers.SGD import SGD
from neural_net_from_scratch.src.losses.sofmax_loss import SoftmaxLoss
import matplotlib.pyplot as plt
from typing import Optional, List
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

def generate_artificial_classification_data_for_softmax_minimazation(n_sampels, n_classes, n_features):
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

def plot_softmax_decision_boundary(X, y, model, weights,  n_classes, title, path = 'neural_net_from_scratch/artifacts/softmax_sgd_decision_boundary_1_3.png'):
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
    plt.savefig(path)
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


def plot_loss_vs_accuracy_train_vs_test(
    train_losses: Optional[List[float]] = None,
    test_losses: Optional[List[float]] = None,
    train_accuracy: Optional[List[float]] = None,
    test_accuracy: Optional[List[float]] = None,
    title: str = "Training Progress",
    path: str = "training_plot.png"
) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    ax2.tick_params(axis='y')

    lines = []
    labels = []

    if train_losses is not None:
        train_loss_line = ax1.plot(train_losses, color='tab:red', label='Train Loss')
        lines.extend(train_loss_line)
        labels.append('Train Loss')

    if test_losses is not None:
        test_loss_line = ax1.plot(test_losses, color='tab:orange', label='Test Loss')
        lines.extend(test_loss_line)
        labels.append('Test Loss')

    if train_accuracy is not None:
        train_acc_line = ax2.plot(train_accuracy, color='tab:blue', label='Train Accuracy')
        lines.extend(train_acc_line)
        labels.append('Train Accuracy')

    if test_accuracy is not None:
        test_acc_line = ax2.plot(test_accuracy, color='tab:green', label='Test Accuracy')
        lines.extend(test_acc_line)
        labels.append('Test Accuracy')

    if lines:
        plt.legend(lines, labels, loc='center left', bbox_to_anchor=(1.15, 0.5))

    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def plot_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_prob: np.ndarray,
    title: str = "Model Performance",
    path: str = "metrics.png"
) -> None:
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = plt.subplot(131)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True)

    ax2 = plt.subplot(132)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Confusion Matrix')

    ax3 = plt.subplot(133)
    ax3.axis('off')
    metrics_text = (
        f"Model Performance Metrics\n\n"
        f"Accuracy: {np.mean(y_pred == y_true):.3f}\n"
        f"AUC-ROC: {roc_auc:.3f}\n\n"
        f"Confusion Matrix:\n"
        f"TN: {cm[0,0]}, FP: {cm[0,1]}\n"
        f"FN: {cm[1,0]}, TP: {cm[1,1]}\n\n"
        f"Precision: {cm[1,1]/(cm[1,1]+cm[0,1]):.3f}\n"
        f"Recall: {cm[1,1]/(cm[1,1]+cm[1,0]):.3f}"
    )
    ax3.text(0.1, 0.5, metrics_text, fontsize=10, va='center')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()