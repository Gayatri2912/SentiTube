"""
Data Preprocessing Module for YouTube Sentiment Analysis
"""
import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
import sys
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import RAW_DATA_PATH, PREPROCESSED_DATA_PATH, PARAMS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


class DataPreprocessing:
    """Class to handle text preprocessing for sentiment analysis"""
    
    def __init__(self):
        self.params = PARAMS['preprocessing']
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        if self.params['remove_urls']:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions
        if self.params['remove_mentions']:
            text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        if self.params['remove_hashtags']:
            text = re.sub(r'#\w+', '', text)
        
        # Remove emojis and special characters
        if self.params['remove_special_chars']:
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', '', text)
        
        # Convert to lowercase
        if self.params['lowercase']:
            text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.params['remove_stopwords']:
            tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatize
        if self.params['lemmatize']:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # Join tokens back to text
        text = ' '.join(tokens)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the entire dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing...")
        
        # Create a copy
        df_processed = df.copy()
        
        # Apply text cleaning
        logger.info("Cleaning text...")
        df_processed['cleaned_comment'] = df_processed['comment'].apply(self.clean_text)
        
        # Filter out very short texts
        min_length = self.params['min_text_length']
        initial_count = len(df_processed)
        df_processed = df_processed[df_processed['cleaned_comment'].str.len() >= min_length]
        removed_count = initial_count - len(df_processed)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} comments with length < {min_length}")
        
        # Reset index
        df_processed = df_processed.reset_index(drop=True)
        
        logger.info(f"Preprocessing completed. Final dataset size: {len(df_processed)}")
        
        return df_processed
    
    def save_data(self, df: pd.DataFrame) -> None:
        """
        Save preprocessed data
        
        Args:
            df: DataFrame to save
        """
        PREPROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PREPROCESSED_DATA_PATH, index=False)
        logger.info(f"Preprocessed data saved to {PREPROCESSED_DATA_PATH}")
    
    def run(self) -> None:
        """Execute the preprocessing pipeline"""
        logger.info("Loading raw data...")
        df = pd.read_csv(RAW_DATA_PATH)
        logger.info(f"Loaded {len(df)} records")
        
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Save preprocessed data
        self.save_data(df_processed)
        
        logger.info("Data preprocessing completed successfully!")


if __name__ == "__main__":
    preprocessing = DataPreprocessing()
    preprocessing.run()
