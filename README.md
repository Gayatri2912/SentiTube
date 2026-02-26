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

Author
--Gayatri Kailaswar
<img width="1902" height="595" alt="Screenshot 2026-02-26 103632" src="https://github.com/user-attachments/assets/c167c2a1-a560-424d-ba76-4de6fe167606" />
<img width="1892" height="623" alt="Screenshot 2026-02-26 103645" src="https://github.com/user-attachments/assets/d58ba426-89ad-45b7-b488-24e9f3b9a902" />
<img width="1919" height="549" alt="Screenshot 2026-02-26 103707" src="https://github.com/user-attachments/assets/09e2e4e5-d7ca-47c1-aa07-58885f0f9194" />


