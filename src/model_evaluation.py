"""
Model Evaluation Module for YouTube Sentiment Analysis
Compares all trained models and selects the best one
"""
import pandas as pd
import numpy as np
import logging
import pickle
import json
from pathlib import Path
import sys
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    FEATURES_PATH, RF_MODEL_PATH, XGB_MODEL_PATH, LGBM_MODEL_PATH,
    BEST_MODEL_PATH, METRICS_PATH, SENTIMENT_LABELS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluation:
    """Class to evaluate and compare trained models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def load_data(self) -> tuple:
        """Load test features"""
        logger.info("Loading test features...")
        with open(FEATURES_PATH, 'rb') as f:
            features_dict = pickle.load(f)
        
        X_test = features_dict['X_test']
        y_test = features_dict['y_test']
        
        logger.info(f"Loaded test data: {X_test.shape}")
        return X_test, y_test
    
    def load_models(self) -> None:
        """Load all trained models"""
        logger.info("Loading trained models...")
        
        model_paths = {
            'Random Forest': RF_MODEL_PATH,
            'XGBoost': XGB_MODEL_PATH,
            'LightGBM': LGBM_MODEL_PATH
        }
        
        for model_name, model_path in model_paths.items():
            try:
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                logger.info(f"Loaded {model_name} model")
            except FileNotFoundError:
                logger.warning(f"Model file not found: {model_path}")
    
    def evaluate_model(self, model_name: str, model, X_test, y_test) -> dict:
        """
        Evaluate a single model
        
        Args:
            model_name: Name of the model
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'roc_auc': roc_auc_score(y_test_bin, y_pred_proba, average='macro', multi_class='ovr')
        }
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Get classification report
        class_report = classification_report(
            y_test, y_pred,
            target_names=[SENTIMENT_LABELS[i] for i in sorted(SENTIMENT_LABELS.keys())],
            output_dict=True
        )
        
        result = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': {
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist()
            }
        }
        
        return result
    
    def compare_models(self) -> str:
        """
        Compare all models and return the best model name
        
        Returns:
            Name of the best model
        """
        logger.info("=" * 60)
        logger.info("Model Comparison Summary")
        logger.info("=" * 60)
        
        comparison_df = pd.DataFrame()
        
        for model_name in self.results.keys():
            metrics = self.results[model_name]['metrics']
            comparison_df[model_name] = pd.Series(metrics)
        
        comparison_df = comparison_df.round(4)
        
        logger.info(f"\n{comparison_df.to_string()}")
        
        # Find best model based on F1 macro score
        best_model_name = comparison_df.loc['f1_macro'].idxmax()
        best_f1_score = comparison_df.loc['f1_macro'].max()
        
        logger.info("=" * 60)
        logger.info(f"Best Model: {best_model_name} (F1 Macro: {best_f1_score:.4f})")
        logger.info("=" * 60)
        
        return best_model_name
    
    def save_best_model(self, best_model_name: str) -> None:
        """
        Save the best model
        
        Args:
            best_model_name: Name of the best model
        """
        logger.info(f"Saving best model: {best_model_name}")
        
        best_model = self.models[best_model_name]
        
        with open(BEST_MODEL_PATH, 'wb') as f:
            pickle.dump(best_model, f)
        
        logger.info(f"Best model saved to {BEST_MODEL_PATH}")
    
    def save_results(self, best_model_name: str) -> None:
        """
        Save all evaluation results
        
        Args:
            best_model_name: Name of the best model
        """
        logger.info("Saving evaluation results...")
        
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON serialization
        results_to_save = {
            'best_model': best_model_name,
            'models': {}
        }
        
        for model_name, result in self.results.items():
            results_to_save['models'][model_name] = {
                'metrics': result['metrics'],
                'confusion_matrix': result['confusion_matrix'],
                'classification_report': result['classification_report']
            }
        
        with open(METRICS_PATH, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        
        logger.info(f"Evaluation results saved to {METRICS_PATH}")
    
    def plot_confusion_matrices(self) -> None:
        """Plot confusion matrices for all models"""
        logger.info("Generating confusion matrix plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            cm = np.array(result['confusion_matrix'])
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[SENTIMENT_LABELS[i] for i in sorted(SENTIMENT_LABELS.keys())],
                yticklabels=[SENTIMENT_LABELS[i] for i in sorted(SENTIMENT_LABELS.keys())],
                ax=axes[idx]
            )
            axes[idx].set_title(f'{model_name}\nConfusion Matrix')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        plot_path = METRICS_PATH.parent / 'confusion_matrices.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrices plot saved to {plot_path}")
        plt.close()
    
    def run(self) -> None:
        """Execute the model evaluation pipeline"""
        logger.info("Starting model evaluation pipeline...")
        
        # Load data and models
        X_test, y_test = self.load_data()
        self.load_models()
        
        # Evaluate all models
        for model_name, model in self.models.items():
            result = self.evaluate_model(model_name, model, X_test, y_test)
            self.results[model_name] = result
        
        # Compare models
        best_model_name = self.compare_models()
        
        # Save best model
        self.save_best_model(best_model_name)
        
        # Save results
        self.save_results(best_model_name)
        
        # Plot confusion matrices
        self.plot_confusion_matrices()
        
        logger.info("Model evaluation completed successfully!")


if __name__ == "__main__":
    evaluation = ModelEvaluation()
    evaluation.run()
