# Common commands for SentiTube project

# Setup
.PHONY: setup
setup:
	python -m venv venv
	.\venv\Scripts\activate && pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Run complete pipeline
.PHONY: pipeline
pipeline:
	python run_pipeline.py

# Data operations
.PHONY: data
data:
	python src/data_ingestion.py
	python src/data_preprocessing.py

# Feature engineering
.PHONY: features
features:
	python src/feature_engineering.py

# Train models
.PHONY: train
train:
	python src/model_training.py

# Evaluate models
.PHONY: evaluate
evaluate:
	python src/model_evaluation.py

# Start MLflow UI
.PHONY: mlflow
mlflow:
	mlflow ui

# Start Flask API
.PHONY: api
api:
	python app/app.py

# Run tests
.PHONY: test
test:
	pytest tests/ -v

# Run tests with coverage
.PHONY: test-cov
test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Code formatting
.PHONY: format
format:
	black src/ app/ tests/

# Linting
.PHONY: lint
lint:
	flake8 src/ app/ tests/
	pylint src/ app/

# Docker build
.PHONY: docker-build
docker-build:
	docker build -t sentitube:latest .

# Docker run
.PHONY: docker-run
docker-run:
	docker run -p 8080:8080 sentitube:latest

# Docker compose
.PHONY: docker-compose
docker-compose:
	docker-compose up --build

# Clean
.PHONY: clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage

# Clean all (including data and models)
.PHONY: clean-all
clean-all: clean
	rm -rf data/raw/*
	rm -rf data/processed/*
	rm -rf models/*
	rm -rf mlruns/*
	rm -rf metrics/*

# Help
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make setup          - Set up virtual environment and install dependencies"
	@echo "  make pipeline       - Run complete ML pipeline"
	@echo "  make data           - Run data ingestion and preprocessing"
	@echo "  make features       - Run feature engineering"
	@echo "  make train          - Train models"
	@echo "  make evaluate       - Evaluate models"
	@echo "  make mlflow         - Start MLflow UI"
	@echo "  make api            - Start Flask API"
	@echo "  make test           - Run tests"
	@echo "  make test-cov       - Run tests with coverage"
	@echo "  make format         - Format code with black"
	@echo "  make lint           - Lint code"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container"
	@echo "  make docker-compose - Run with docker-compose"
	@echo "  make clean          - Clean temporary files"
	@echo "  make clean-all      - Clean everything including data and models"
