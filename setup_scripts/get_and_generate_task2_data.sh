#!/bin/bash

set -e

BASE_DIR=$(pwd)
MNIST_DIR="RNNBasics/data/mnist"
STOCK_DIR="RNNBasics/data/sp500"
SYNTHETIC_DIR="RNNBasics/data/synthetic_data"

mkdir -p $MNIST_DIR
mkdir -p $STOCK_DIR
mkdir -p $SYNTHETIC_DIR

echo "Downloading MNIST dataset..."
wget --no-check-certificate -P $MNIST_DIR https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
wget --no-check-certificate -P $MNIST_DIR https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
wget --no-check-certificate -P $MNIST_DIR https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
wget --no-check-certificate -P $MNIST_DIR https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
gunzip -f $MNIST_DIR/*.gz

echo "Attempting to download S&P 500 dataset..."
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Warning: Kaggle credentials not found. Trying direct URL download..."
    mkdir -p $STOCK_DIR
    wget -O $STOCK_DIR/sp500_stocks.csv https://raw.githubusercontent.com/gauravmehta13/sp-500-stock-prices/master/sp500_stocks.csv || {
        echo "Direct download failed. Please download manually from Kaggle:"
        echo "To download S&P 500 data later:"
        echo "1. Create a Kaggle account at https://www.kaggle.com"
        echo "2. Go to 'Account' -> 'Create New API Token'"
        echo "3. Place the downloaded kaggle.json in ~/.kaggle/"
        echo "4. Run 'chmod 600 ~/.kaggle/kaggle.json'"
        echo "5. Rerun 'make get_task2_data'"
    }
else
    kaggle datasets download gauravmehta13/sp-500-stock-prices -p $STOCK_DIR
    unzip -o $STOCK_DIR/sp-500-stock-prices.zip -d $STOCK_DIR
    rm $STOCK_DIR/sp-500-stock-prices.zip
fi

echo "Generating synthetic data..."
cd $BASE_DIR
python $BASE_DIR/setup_scripts/setup_synthetic_data_task_2.py  # Updated filename here

echo "Task 2 data preparation completed!"
if [ ! -f $STOCK_DIR/sp500_stocks.csv ]; then
    echo "Note: S&P 500 dataset was not downloaded. See instructions above to download it later."
fi