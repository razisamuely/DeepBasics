SHELL := /bin/bash
PYTHONPATH := $(shell pwd)
SCRIPTS_DIR := setup_scripts
MNIST_DIR := RNNBasics/data
DATA_DIR := neural_net_from_scratch/data
STOCK_DIR := RNNBasics/data/sp500
SYNTHETIC_DIR := RNNBasics/data/synthetic_data

all: venv requirements python_path get_data get_mnist get_sp500 generate_synthetic

venv:
	python3 -m venv venv
	. venv/bin/activate && pip install uv

requirements: venv
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install kaggle h5py

python_path:
	@echo "Setting PYTHONPATH to: $(PYTHONPATH)"
	@echo 'export PYTHONPATH=$(PYTHONPATH)' >> venv/bin/activate  # Embed in venv activation script

get_data:
	. venv/bin/activate && python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1s06UaGqogkFmrhnuRK7Tm9fBNi6NYug0', '$(DATA_DIR)/data.zip', quiet=False)"
	mkdir -p $(DATA_DIR)
	unzip -o $(DATA_DIR)/data.zip -d $(DATA_DIR)
	rm $(DATA_DIR)/data.zip

get_mnist:
	mkdir -p $(MNIST_DIR)
	wget --no-check-certificate -P $(MNIST_DIR) https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
	wget --no-check-certificate -P $(MNIST_DIR) https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
	wget --no-check-certificate -P $(MNIST_DIR) https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
	wget --no-check-certificate -P $(MNIST_DIR) https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
	gunzip $(MNIST_DIR)/*.gz

get_sp500:
	mkdir -p $(STOCK_DIR)
	@echo "Please ensure you have kaggle.json in ~/.kaggle/ directory"
	. venv/bin/activate && kaggle datasets download gauravmehta13/sp-500-stock-prices -p $(STOCK_DIR)
	unzip -o $(STOCK_DIR)/sp-500-stock-prices.zip -d $(STOCK_DIR)
	rm $(STOCK_DIR)/sp-500-stock-prices.zip

generate_synthetic:
	mkdir -p $(SYNTHETIC_DIR)
	. venv/bin/activate && python RNNBasics/data/setup_synthetic_data.py

clean:
	rm -rf venv
	rm -rf $(DATA_DIR)/*
	rm -rf $(MNIST_DIR)/*
	rm -rf $(STOCK_DIR)/*
	rm -rf $(SYNTHETIC_DIR)/*
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

.PHONY: all venv requirements python_path get_data get_mnist get_sp500 generate_synthetic clean
.EXPORT_ALL_VARIABLES: