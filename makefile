SHELL := /bin/bash
PYTHONPATH := $(shell pwd)
SCRIPTS_DIR := setup_scripts
DATA_DIR := neural_net_from_scratch/data

all: venv requirements python_path get_data get_task2_data

venv:
	python3 -m venv venv
	. venv/bin/activate && pip install uv

requirements: venv
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install kaggle h5py

python_path:
	@echo "Setting PYTHONPATH to: $(PYTHONPATH)"
	@echo 'export PYTHONPATH=$(PYTHONPATH)' >> venv/bin/activate

get_data:
	mkdir -p $(DATA_DIR)
	. venv/bin/activate && python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1s06UaGqogkFmrhnuRK7Tm9fBNi6NYug0', '$(DATA_DIR)/data.zip', quiet=False)"
	unzip -o $(DATA_DIR)/data.zip -d $(DATA_DIR)
	rm $(DATA_DIR)/data.zip

get_task2_data:
	chmod +x $(SCRIPTS_DIR)/get_and_generate_task2_data.sh
	. venv/bin/activate && $(SCRIPTS_DIR)/get_and_generate_task2_data.sh

clean:
	rm -rf venv
	rm -rf $(DATA_DIR)/*
	rm -rf RNNBasics/data/*
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

.PHONY: all venv requirements python_path get_data get_task2_data clean
.EXPORT_ALL_VARIABLES: