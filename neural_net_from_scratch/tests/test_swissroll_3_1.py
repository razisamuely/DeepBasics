import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import scipy
import tqdm
from neural_net_from_scratch.src.optimizers.SGD import SGD
from neural_net_from_scratch.src.losses.sofmax_loss import SoftmaxLoss
from neural_net_from_scratch.src.models.neural_network import DynamicNeuralNetwork
from neural_net_from_scratch.tests.utils import (
    plot_loss_vs_accuracy_train_vs_test,
    plot_metrics
)

N_FEATURES = 2
N_CLASSES = 2
HIDDEN_DIM = 32
LEARNING_RATE = 1e-2
N_EPOCHS = 2000
BATCH_SIZE = 32
N_ITERATIONS = 500
DECAY = 0.999


def train_network(
    network: DynamicNeuralNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    learning_rate: float = 1e-3,
    n_epochs: int = 2000
) -> Tuple[List[float], List[float], List[float], List[float]]:
    losses, train_accuracies, test_accuracies, test_losses = [], [], [], []
    optimizer = SGD(model=network, learning_rate=learning_rate)
    softmax = SoftmaxLoss()
    clip_threshold = 1.0
    
    for epoch in tqdm.tqdm(range(n_epochs)):
        shuffle_idx = np.random.permutation(X_train.shape[1])
        X_train_shuffled = X_train[:, shuffle_idx]
        y_train_shuffled = y_train[:, shuffle_idx]
        epoch_losses, epoch_accuracies = [], []
        for i in range(0, X_train.shape[1], BATCH_SIZE):
            X_batch = X_train_shuffled[:, i:i + BATCH_SIZE]
            y_batch = y_train_shuffled[:, i:i + BATCH_SIZE]
            
            outputs = network.forward(X_batch)
            loss = softmax.loss(np.argmax(outputs, axis=0), y_batch)
            grad = softmax.loss_gradient(outputs, y_batch)
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm > clip_threshold:
                grad = grad * (clip_threshold / grad_norm)
            
            network.backward(grad)
            optimizer.step()
            
            epoch_losses.append(loss)
            pred_classes = np.argmax(outputs, axis=0)
            true_classes = np.argmax(y_batch, axis=0)
            epoch_accuracies.append(np.mean(pred_classes == true_classes))
        
        avg_loss = np.mean(epoch_losses)
        avg_accuracy = np.mean(epoch_accuracies)
        losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)
        
        test_outputs = network.forward(X_test)
        test_loss = softmax.loss(np.argmax(test_outputs, axis=0), y_test)
        test_pred = np.argmax(test_outputs, axis=0)
        test_true = np.argmax(y_test, axis=0)
        test_accuracy = np.mean(test_pred == test_true)
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Train - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
            print(f"  Test  - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
            print(f"  Learning rate: {learning_rate:.6f}")
        
        learning_rate *= DECAY
        optimizer.learning_rate = learning_rate
    
    return losses, train_accuracies, test_accuracies, test_losses, network

def load_and_preprocess_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = scipy.io.loadmat(data_path)
    X_train, y_train = data['Yt'].T, data['Ct'].T
    X_val, y_val = data['Yv'].T, data['Cv'].T
    
    train_mean, train_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    
    return X_train, y_train, X_val, y_val

def main() -> None:
    layer_configs = [
        {"type": "layer", "input_dim": N_FEATURES, "output_dim": 64, "activation": "relu"},
        {"type": "layer", "input_dim": 64, "output_dim": 32, "activation": "relu"},
        {"type": "layer", "input_dim": 32, "output_dim": N_CLASSES, "activation": "softmax"},
    ]
    
    X_train, y_train, X_val, y_val = load_and_preprocess_data('neural_net_from_scratch/data/SwissRollData.mat')
    network = DynamicNeuralNetwork(layer_configs)
    
    losses, train_accuracies, test_accuracies, test_losses, network = train_network(
        network=network,
        X_train=X_train.T,
        y_train=y_train.T,
        X_test=X_val.T,
        y_test=y_val.T,
        n_epochs=N_ITERATIONS,
        learning_rate=LEARNING_RATE
    )

    plot_loss_vs_accuracy_train_vs_test(
        train_losses=losses,
        test_losses=test_losses,
        train_accuracy=train_accuracies,
        test_accuracy=test_accuracies,
        title=f'Training Progress (LR={LEARNING_RATE}, Hidden={HIDDEN_DIM})',
        path='neural_net_from_scratch/artifacts/swissroll_training_curves_3_1.png'
    )

    test_outputs = network.forward(X_val.T)
    softmax = SoftmaxLoss()
    test_probs = softmax.sofotmax(test_outputs)
    test_pred = np.argmax(test_outputs, axis=0)
    test_true = np.argmax(y_val.T, axis=0)

    plot_metrics(
        y_true=test_true,
        y_pred=test_pred,
        y_pred_prob=test_probs.T,
        title=f'Final Model Performance (LR={LEARNING_RATE}, Hidden={HIDDEN_DIM})',
        path='neural_net_from_scratch/artifacts/swissroll_auc_cm_matrix_3_1.png'
    )

if __name__ == "__main__":
    main()
