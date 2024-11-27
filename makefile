SHELL := /bin/bash
PYTHONPATH := $(shell pwd)
SCRIPTS_DIR := setup_scripts

all: venv requirements activate python_path get_data

venv:
	python3 -m venv venv
	. venv/bin/activate && pip install uv

requirements: venv
	. venv/bin/activate && uv pip install -r requirements.txt

activate:
	@echo "To activate the virtual environment, run: source venv/bin/activate"

python_path:
	@echo "Setting PYTHONPATH to: $(PYTHONPATH)"
	@export PYTHONPATH=$(PYTHONPATH)

get_data:
	. venv/bin/activate && python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1s06UaGqogkFmrhnuRK7Tm9fBNi6NYug0', 'neural_net_from_scratch/data/data.zip', quiet=False)"
	mkdir -p neural_net_from_scratch/data
	unzip -o neural_net_from_scratch/data/data.zip -d neural_net_from_scratch/data
	rm neural_net_from_scratch/data/data.zip
	
clean:
	rm -rf venv
	rm -rf neural_net_from_scratch/data/*
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

.PHONY: all venv requirements activate python_path get_data clean
.EXPORT_ALL_VARIABLES: