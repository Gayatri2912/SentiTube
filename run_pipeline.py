"""
Run the complete pipeline for SentiTube
This script executes all steps from data ingestion to model evaluation
"""
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def run_pipeline():
    """Execute the complete ML pipeline"""
    
    logger.info("="*70)
    logger.info("STARTING SENTITUBE PIPELINE")
    logger.info("="*70)
    
    try:
        # Step 1: Data Ingestion
        logger.info("\n" + "="*70)
        logger.info("STEP 1: Data Ingestion")
        logger.info("="*70)
        from src.data_ingestion import DataIngestion
        ingestion = DataIngestion()
        ingestion.run()
        logger.info("✓ Data ingestion completed successfully")
        
        # Step 2: Data Preprocessing
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Data Preprocessing")
        logger.info("="*70)
        from src.data_preprocessing import DataPreprocessing
        preprocessing = DataPreprocessing()
        preprocessing.run()
        logger.info("✓ Data preprocessing completed successfully")
        
        # Step 3: Feature Engineering
        logger.info("\n" + "="*70)
        logger.info("STEP 3: Feature Engineering")
        logger.info("="*70)
        from src.feature_engineering import FeatureEngineering
        feature_eng = FeatureEngineering()
        feature_eng.run()
        logger.info("✓ Feature engineering completed successfully")
        
        # Step 4: Model Training
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Model Training")
        logger.info("="*70)
        from src.model_training import ModelTraining
        trainer = ModelTraining()
        trainer.run()
        logger.info("✓ Model training completed successfully")
        
        # Step 5: Model Evaluation
        logger.info("\n" + "="*70)
        logger.info("STEP 5: Model Evaluation")
        logger.info("="*70)
        from src.model_evaluation import ModelEvaluation
        evaluation = ModelEvaluation()
        evaluation.run()
        logger.info("✓ Model evaluation completed successfully")
        
        # Pipeline completed
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info("\nNext steps:")
        logger.info("1. View MLflow experiments: mlflow ui")
        logger.info("2. View evaluation metrics: cat metrics/evaluation_results.json")
        logger.info("3. Start the API: python app/app.py")
        logger.info("4. Run with Docker: docker-compose up --build")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed with error: {e}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
