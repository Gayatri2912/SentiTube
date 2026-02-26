<div align="center">

# 🎬 SentiTube

### AI-Powered YouTube Sentiment Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Pipeline-purple.svg)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Transform YouTube comments into actionable insights with cutting-edge NLP and Machine Learning**

[Quick Start](#-quick-start) • [Features](#-features) • [Documentation](GETTING_STARTED.md) • [API Docs](API_DOCS.md)

</div>

---

## 📊 Project Overview

**SentiTube** is an end-to-end MLOps pipeline for analyzing sentiment in YouTube comments using Natural Language Processing (NLP) and Machine Learning. This production-ready platform demonstrates best practices in ML experimentation, deployment, and monitoring.

## 🎯 Features

- **Multiple ML Models**: Random Forest, XGBoost, and LightGBM
- **Experiment Tracking**: MLflow for tracking experiments, parameters, and metrics
- **Data Version Control**: DVC for versioning datasets and models
- **Imbalanced Data Handling**: SMOTE for addressing class imbalance
- **Feature Engineering**: TF-IDF vectorization
- **Containerization**: Docker for consistent deployment
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Interactive API**: Flask web application for predictions

## 🏗️ Project Structure

```
sentitube/
├── data/                      # Data directory (DVC tracked)
│   ├── raw/                   # Raw YouTube comments data
│   └── processed/             # Processed and feature-engineered data
├── notebooks/                 # Jupyter notebooks for EDA
│   └── EDA.ipynb             # Exploratory Data Analysis
├── src/                       # Source code
│   ├── data_ingestion.py     # Data loading and collection
│   ├── data_preprocessing.py # Text preprocessing
│   ├── feature_engineering.py # TF-IDF and feature creation
│   ├── model_training.py     # Model training with MLflow
│   ├── model_evaluation.py   # Model comparison and evaluation
│   └── prediction.py         # Prediction pipeline
├── app/                       # Flask web application
│   ├── app.py                # Web API
│   └── templates/            # HTML templates
├── models/                    # Saved models
├── experiments/              # Experiment configurations
├── tests/                    # Unit tests
├── .github/workflows/        # CI/CD pipelines
├── Dockerfile               # Docker container definition
├── docker-compose.yml       # Multi-container orchestration
├── requirements.txt         # Python dependencies
├── dvc.yaml                 # DVC pipeline definition
├── params.yaml              # Hyperparameters and configurations
└── README.md                # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (optional)
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd youtube
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize DVC**
```bash
dvc init
dvc pull  # If data is already versioned
```

## 📈 Usage

### 1. Data Ingestion
```bash
python src/data_ingestion.py
```

### 2. Run EDA
```bash
jupyter notebook notebooks/EDA.ipynb
```

### 3. Data Preprocessing
```bash
python src/data_preprocessing.py
```

### 4. Feature Engineering
```bash
python src/feature_engineering.py
```

### 5. Train Models with MLflow
```bash
python src/model_training.py
```

View experiments in MLflow UI:
```bash
mlflow ui
```
Navigate to http://localhost:5000

### 6. Evaluate Models
```bash
python src/model_evaluation.py
```

### 7. Run Prediction API
```bash
python app/app.py
```
Navigate to http://localhost:8080

### 8. Run with Docker
```bash
docker-compose up --build
```

## 🔬 Model Experimentation

The project experiments with three models:

1. **Random Forest**: Baseline ensemble model
2. **XGBoost**: Gradient boosting with regularization
3. **LightGBM**: Fast gradient boosting (best performer)

All experiments are tracked in MLflow with:
- Hyperparameters
- Metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Model artifacts
- Visualizations

## 🛠️ Technologies Used

- **ML/DL**: scikit-learn, XGBoost, LightGBM
- **NLP**: NLTK, spaCy, TF-IDF
- **Experiment Tracking**: MLflow
- **Version Control**: DVC, Git
- **Data Handling**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Imbalance Handling**: imbalanced-learn (SMOTE)
- **Web Framework**: Flask
- **Containerization**: Docker
- **CI/CD**: GitHub Actions

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD | TBD | TBD |

*Results updated after training*

## 🔄 CI/CD Pipeline

GitHub Actions workflow includes:
1. Code linting (flake8, black)
2. Unit tests (pytest)
3. Model training and evaluation
4. Docker image build
5. Performance reporting

## 📝 License

MIT License

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue.
