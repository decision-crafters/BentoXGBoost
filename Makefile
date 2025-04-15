.PHONY: setup clean venv install run-default run-github run-web run-crawl serve serve-model list-projects serve-project run-project help

# Variables
VENV_NAME := venv
PYTHON := python3.11
PIP := $(VENV_NAME)/bin/pip
PYTHON_VENV := $(VENV_NAME)/bin/python
BENTOML := $(VENV_NAME)/bin/bentoml

# Default target
.DEFAULT_GOAL := help

# Help target
help:
	@echo "Available targets:"
	@echo "  make setup        - Create virtual environment and install dependencies"
	@echo "  make clean        - Remove virtual environment and cached files"
	@echo "  make venv         - Create virtual environment"
	@echo "  make install      - Install dependencies"
	@echo "  make run-default  - Run model training with default dataset"
	@echo "  make run-github   - Run model training with GitHub repository"
	@echo "  make run-web      - Run model training with web page"
	@echo "  make run-crawl    - Run model training with crawled website"
	@echo "  make serve        - Start BentoML service"
	@echo "  make serve-model MODEL=name - Start BentoML service with a specific model"
	@echo "  make list-projects - List all available projects from config.yaml"
	@echo "  make serve-project PROJECT=name - Start BentoML service with a specific project"
	@echo "  make run-project PROJECT=name - Run model training for a specific project"

# Setup target
setup: venv install

# Clean target
clean:
	rm -rf $(VENV_NAME)
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Virtual environment target
venv:
	@echo "Creating virtual environment..."
	@if command -v $(PYTHON) > /dev/null; then \
		$(PYTHON) -m venv $(VENV_NAME); \
	else \
		echo "Python 3.11 is not installed. Please install it first."; \
		exit 1; \
	fi
	@echo "Virtual environment created."

# Install dependencies target
install: venv
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@if [ "$(shell uname)" = "Darwin" ]; then \
		echo "macOS detected. Installing libomp for XGBoost..."; \
		if command -v brew > /dev/null; then \
			brew install libomp || echo "Warning: Failed to install libomp. You may need to install it manually."; \
		else \
			echo "Warning: Homebrew not found. Please install libomp manually for XGBoost to work on macOS."; \
		fi \
	fi
	@echo "Dependencies installed."

# Run model training with default dataset
run-default: install
	@echo "Running model training with default dataset..."
	$(PYTHON_VENV) save_model.py --use-default-dataset --model-name default_model

# Run model training with GitHub repository
run-github: install
	@echo "Running model training with GitHub repository..."
	$(PYTHON_VENV) save_model.py --github-repo-url https://github.com/KolbySisk/next-supabase-stripe-starter/archive/refs/heads/main.zip --model-name github_model

# Run model training with web page
run-web: install
	@echo "Running model training with web page..."
	$(PYTHON_VENV) save_model.py --web-url https://www.cancer.org/cancer/types/breast-cancer/about/what-is-breast-cancer.html --model-name web_model

# Run model training with crawled website
run-crawl: install
	@echo "Running model training with crawled website..."
	$(PYTHON_VENV) save_model.py --crawl-url https://www.cancer.org/cancer/types/breast-cancer/ --max-pages 5 --model-name crawled_model

# Start BentoML service
serve: install
	@echo "Starting BentoML service..."
	$(BENTOML) serve .

# Start BentoML service with a specific model
serve-model: install
	@echo "Starting BentoML service with model $(MODEL)..."
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL parameter is required. Usage: make serve-model MODEL=model_name"; \
		exit 1; \
	fi
	MODEL_NAME=$(MODEL) $(BENTOML) serve .

# List all available projects from config.yaml
list-projects: install
	@echo "Listing all available projects..."
	$(PYTHON_VENV) save_model.py --list-projects

# Start BentoML service with a specific project
serve-project: install
	@echo "Starting BentoML service with project $(PROJECT)..."
	@if [ -z "$(PROJECT)" ]; then \
		echo "Error: PROJECT parameter is required. Usage: make serve-project PROJECT=project_name"; \
		exit 1; \
	fi
	BENTO_PROJECT=$(PROJECT) $(BENTOML) serve .

# Run model training for a specific project
run-project: install
	@echo "Running model training for project $(PROJECT)..."
	@if [ -z "$(PROJECT)" ]; then \
		echo "Error: PROJECT parameter is required. Usage: make run-project PROJECT=project_name"; \
		exit 1; \
	fi
	$(PYTHON_VENV) save_model.py --project $(PROJECT)
