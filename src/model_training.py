"""
Model Training Module for YouTube Sentiment Analysis
Trains multiple models (Random Forest, XGBoost, LightGBM) with MLflow tracking
"""
import pandas as pd
import numpy as np
import logging
import pickle
from pathlib import Path
import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    FEATURES_PATH, RF_MODEL_PATH, XGB_MODEL_PATH, LGBM_MODEL_PATH,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, PARAMS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTraining:
    """Class to handle model training with MLflow tracking"""
    
    def __init__(self):
        self.rf_params = PARAMS['random_forest']
        self.xgb_params = PARAMS['xgboost']
        self.lgbm_params = PARAMS['lightgbm']
        
        # Set up MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
    def load_features(self) -> tuple:
        """Load preprocessed features"""
        logger.info("Loading features...")
        with open(FEATURES_PATH, 'rb') as f:
            features_dict = pickle.load(f)
        
        X_train = features_dict['X_train']
        X_test = features_dict['X_test']
        y_train = features_dict['y_train']
        y_test = features_dict['y_test']
        
        logger.info(f"Loaded features - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba) -> dict:
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        # For multiclass ROC-AUC, we need one-vs-rest
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'roc_auc_ovr': roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr')
        }
        
        return metrics
    
    def train_random_forest(self, X_train, X_test, y_train, y_test) -> None:
        """Train Random Forest model with MLflow tracking"""
        logger.info("=" * 50)
        logger.info("Training Random Forest model...")
        
        with mlflow.start_run(run_name="Random_Forest"):
            # Log parameters
            mlflow.log_params(self.rf_params)
            
            # Train model
            model = RandomForestClassifier(**self.rf_params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Save model locally
            with open(RF_MODEL_PATH, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Random Forest Metrics: {metrics}")
            logger.info(f"Model saved to {RF_MODEL_PATH}")
    
    def train_xgboost(self, X_train, X_test, y_train, y_test) -> None:
        """Train XGBoost model with MLflow tracking"""
        logger.info("=" * 50)
        logger.info("Training XGBoost model...")
        
        with mlflow.start_run(run_name="XGBoost"):
            # Log parameters
            mlflow.log_params(self.xgb_params)
            
            # Train model
            model = XGBClassifier(**self.xgb_params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Save model locally
            with open(XGB_MODEL_PATH, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"XGBoost Metrics: {metrics}")
            logger.info(f"Model saved to {XGB_MODEL_PATH}")
    
    def train_lightgbm(self, X_train, X_test, y_train, y_test) -> None:
        """Train LightGBM model with MLflow tracking"""
        logger.info("=" * 50)
        logger.info("Training LightGBM model...")
        
        with mlflow.start_run(run_name="LightGBM"):
            # Log parameters
            mlflow.log_params(self.lgbm_params)
            
            # Train model
            model = LGBMClassifier(**self.lgbm_params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Save model locally
            with open(LGBM_MODEL_PATH, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"LightGBM Metrics: {metrics}")
            logger.info(f"Model saved to {LGBM_MODEL_PATH}")
    
    def run(self) -> None:
        """Execute the model training pipeline"""
        logger.info("Starting model training pipeline...")
        
        # Load features
        X_train, X_test, y_train, y_test = self.load_features()
        
        # Train all models
        self.train_random_forest(X_train, X_test, y_train, y_test)
        self.train_xgboost(X_train, X_test, y_train, y_test)
        self.train_lightgbm(X_train, X_test, y_train, y_test)
        
        logger.info("=" * 50)
        logger.info("Model training completed successfully!")
        logger.info("View experiments with: mlflow ui")


if __name__ == "__main__":
    trainer = ModelTraining()
    trainer.run()
