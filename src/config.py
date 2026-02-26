"""
Configuration module for SentiTube
"""
import os
import yaml
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "metrics"
MLRUNS_DIR = ROOT_DIR / "mlruns"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "raw").mkdir(exist_ok=True)
(DATA_DIR / "processed").mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)
MLRUNS_DIR.mkdir(exist_ok=True)


def load_params(params_path: str = None) -> dict:
    """
    Load parameters from params.yaml file
    
    Args:
        params_path: Path to params.yaml file
        
    Returns:
        Dictionary containing all parameters
    """
    if params_path is None:
        params_path = ROOT_DIR / "params.yaml"
    
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    
    return params


# Load parameters
PARAMS = load_params()

# Data paths
RAW_DATA_PATH = DATA_DIR / "raw" / "youtube_comments.csv"
PREPROCESSED_DATA_PATH = DATA_DIR / "processed" / "preprocessed_data.csv"
FEATURES_PATH = DATA_DIR / "processed" / "features.pkl"

# Model paths
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
RF_MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
LGBM_MODEL_PATH = MODELS_DIR / "lightgbm_model.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

# Metrics path
METRICS_PATH = METRICS_DIR / "evaluation_results.json"

# MLflow configuration
MLFLOW_TRACKING_URI = f"file:///{str(MLRUNS_DIR).replace(chr(92), '/')}"
MLFLOW_EXPERIMENT_NAME = PARAMS.get("mlflow", {}).get("experiment_name", "sentitube")

# Sentiment labels
SENTIMENT_LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# API configuration
API_HOST = PARAMS.get("api", {}).get("host", "0.0.0.0")
API_PORT = PARAMS.get("api", {}).get("port", 8080)
API_DEBUG = PARAMS.get("api", {}).get("debug", False)
