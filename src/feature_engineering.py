"""
Feature Engineering Module for YouTube Sentiment Analysis
"""
import pandas as pd
import numpy as np
import logging
import pickle
from pathlib import Path
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import PREPROCESSED_DATA_PATH, FEATURES_PATH, VECTORIZER_PATH, PARAMS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Class to handle feature engineering for sentiment analysis"""
    
    def __init__(self):
        self.fe_params = PARAMS['feature_engineering']
        self.data_params = PARAMS['data_ingestion']
        self.imbalance_params = PARAMS['imbalance']
        self.vectorizer = None
        
    def extract_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract additional text-based features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        logger.info("Extracting additional features...")
        
        df_features = df.copy()
        
        # Text length
        if 'text_length' in self.fe_params['additional_features']:
            df_features['text_length'] = df_features['cleaned_comment'].str.len()
        
        # Word count
        if 'word_count' in self.fe_params['additional_features']:
            df_features['word_count'] = df_features['cleaned_comment'].str.split().str.len()
        
        # Exclamation marks count
        if 'exclamation_count' in self.fe_params['additional_features']:
            df_features['exclamation_count'] = df_features['comment'].str.count('!')
        
        # Question marks count
        if 'question_count' in self.fe_params['additional_features']:
            df_features['question_count'] = df_features['comment'].str.count('\?')
        
        # Uppercase ratio (in original comment)
        if 'uppercase_ratio' in self.fe_params['additional_features']:
            df_features['uppercase_ratio'] = df_features['comment'].apply(
                lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1)
            )
        
        logger.info(f"Extracted {len(self.fe_params['additional_features'])} additional features")
        
        return df_features
    
    def create_tfidf_features(self, X_train: pd.Series, X_test: pd.Series) -> tuple:
        """
        Create TF-IDF features
        
        Args:
            X_train: Training text data
            X_test: Test text data
            
        Returns:
            Tuple of (X_train_tfidf, X_test_tfidf)
        """
        logger.info("Creating TF-IDF features...")
        
        tfidf_params = self.fe_params['tfidf']
        
        self.vectorizer = TfidfVectorizer(
            max_features=tfidf_params['max_features'],
            ngram_range=tuple(tfidf_params['ngram_range']),
            max_df=tfidf_params['max_df'],
            min_df=tfidf_params['min_df']
        )
        
        # Fit and transform training data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        # Transform test data
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        logger.info(f"TF-IDF vocabulary size: {len(self.vectorizer.vocabulary_)}")
        logger.info(f"TF-IDF feature matrix shape - Train: {X_train_tfidf.shape}, Test: {X_test_tfidf.shape}")
        
        return X_train_tfidf, X_test_tfidf
    
    def combine_features(self, tfidf_features, additional_features) -> np.ndarray:
        """
        Combine TF-IDF features with additional features
        
        Args:
            tfidf_features: TF-IDF feature matrix
            additional_features: Additional features DataFrame
            
        Returns:
            Combined feature matrix
        """
        from scipy.sparse import hstack
        
        # Convert additional features to array
        add_features_array = additional_features.values
        
        # Combine features
        combined = hstack([tfidf_features, add_features_array])
        
        return combined
    
    def handle_imbalance(self, X_train, y_train) -> tuple:
        """
        Handle class imbalance using SMOTE
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        logger.info("Handling class imbalance with SMOTE...")
        logger.info(f"Original class distribution:\n{pd.Series(y_train).value_counts()}")
        
        smote = SMOTE(
            sampling_strategy=self.imbalance_params['sampling_strategy'],
            k_neighbors=self.imbalance_params['k_neighbors'],
            random_state=self.data_params['random_state']
        )
        
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        logger.info(f"Resampled class distribution:\n{pd.Series(y_resampled).value_counts()}")
        logger.info(f"Training set size after SMOTE: {X_resampled.shape[0]}")
        
        return X_resampled, y_resampled
    
    def save_features(self, X_train, X_test, y_train, y_test) -> None:
        """
        Save features and vectorizer
        
        Args:
            X_train, X_test, y_train, y_test: Train-test split data
        """
        logger.info("Saving features and vectorizer...")
        
        # Save features
        features_dict = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FEATURES_PATH, 'wb') as f:
            pickle.dump(features_dict, f)
        logger.info(f"Features saved to {FEATURES_PATH}")
        
        # Save vectorizer
        with open(VECTORIZER_PATH, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        logger.info(f"Vectorizer saved to {VECTORIZER_PATH}")
    
    def run(self) -> None:
        """Execute the feature engineering pipeline"""
        logger.info("Loading preprocessed data...")
        df = pd.read_csv(PREPROCESSED_DATA_PATH)
        logger.info(f"Loaded {len(df)} records")
        
        # Extract additional features
        df_with_features = self.extract_additional_features(df)
        
        # Split data
        logger.info("Splitting data into train and test sets...")
        train_df, test_df = train_test_split(
            df_with_features,
            test_size=self.data_params['test_size'],
            random_state=self.data_params['random_state'],
            stratify=df_with_features['sentiment'] if self.data_params['stratify'] else None
        )
        
        logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        # Create TF-IDF features
        X_train_tfidf, X_test_tfidf = self.create_tfidf_features(
            train_df['cleaned_comment'],
            test_df['cleaned_comment']
        )
        
        # Get additional features
        feature_cols = self.fe_params['additional_features']
        train_add_features = train_df[feature_cols]
        test_add_features = test_df[feature_cols]
        
        # Combine features
        logger.info("Combining all features...")
        X_train = self.combine_features(X_train_tfidf, train_add_features)
        X_test = self.combine_features(X_test_tfidf, test_add_features)
        
        y_train = train_df['sentiment'].values
        y_test = test_df['sentiment'].values
        
        # Handle imbalance
        X_train_balanced, y_train_balanced = self.handle_imbalance(X_train, y_train)
        
        # Save features
        self.save_features(X_train_balanced, X_test, y_train_balanced, y_test)
        
        logger.info("Feature engineering completed successfully!")
        logger.info(f"Final feature dimensions - Train: {X_train_balanced.shape}, Test: {X_test.shape}")


if __name__ == "__main__":
    feature_eng = FeatureEngineering()
    feature_eng.run()
