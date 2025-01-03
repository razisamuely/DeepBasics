import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
N_EPOCHS = 30
N_TRIALS = 30

HIDDEN_SIZE_MIN = 32
HIDDEN_SIZE_MAX = 256
LEARNING_RATE_MIN = 1e-4
LEARNING_RATE_MAX = 1e-2
GRAD_CLIP_MIN = 0.1
GRAD_CLIP_MAX = 10.0

INPUT_SIZE = 28
NUM_CLASSES = 10
VAL_SPLIT = 0.2

TRAIN_DATA_PATH = 'RNNBasics/data/mnist/train-images-idx3-ubyte'
TRAIN_LABELS_PATH = 'RNNBasics/data/mnist/train-labels-idx1-ubyte'
TEST_DATA_PATH = 'RNNBasics/data/mnist/t10k-images-idx3-ubyte'
TEST_LABELS_PATH = 'RNNBasics/data/mnist/t10k-labels-idx1-ubyte'

ARTIFACTS_DIR = 'RNNBasics/artifacts/mnist_3_2_1'