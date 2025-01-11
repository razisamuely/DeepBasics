from pathlib import Path
import torch

# Constants for file paths
RESULTS_DIR = Path("RNNBasics/artifacts/results_3_1_2")

# Training defaults
DEFAULT_BATCH_SIZE = 16
DEFAULT_N_EPOCHS = 1000
DEFAULT_N_TRIALS = 4
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_HIDDEN_SIZE = 49
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_GRAD_CLIP = 1.0

# Hyperparameter search ranges
DEFAULT_HIDDEN_SIZE_MIN = 32
DEFAULT_HIDDEN_SIZE_MAX = 49
DEFAULT_LR_MIN = 1e-4
DEFAULT_LR_MAX = 1e-2
DEFAULT_GRAD_CLIP_MIN = 0.1
DEFAULT_GRAD_CLIP_MAX = 5.0
DEFAULT_NUM_LAYERS_MIN = 1
DEFAULT_NUM_LAYERS_MAX = 20

# Model configuration
INPUT_SIZE = 1
N_EXAMPLES_TO_PLOT = 2

# Optimizer and loss options
DEFAULT_OPTIMIZER = 'adam'
DEFAULT_LOSS = 'mse'
AVAILABLE_OPTIMIZERS = ['adam', 'sgd', 'adamw']
AVAILABLE_LOSSES = ['mse', 'l1', 'smoothl1']