"""
Prediction Module for YouTube Sentiment Analysis
"""
import pickle
import logging
from pathlib import Path
import sys
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import BEST_MODEL_PATH, VECTORIZER_PATH, SENTIMENT_LABELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentPredictor:
    """Class to handle sentiment prediction for new comments"""
    
    def __init__(self, model_path=None, vectorizer_path=None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
            vectorizer_path: Path to fitted vectorizer
        """
        self.model_path = model_path or BEST_MODEL_PATH
        self.vectorizer_path = vectorizer_path or VECTORIZER_PATH
        
        self.model = None
        self.vectorizer = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        self.load_artifacts()
    
    def load_artifacts(self) -> None:
        """Load model and vectorizer"""
        logger.info("Loading model and vectorizer...")
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from {self.model_path}")
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise
        
        try:
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info(f"Vectorizer loaded from {self.vectorizer_path}")
        except FileNotFoundError:
            logger.error(f"Vectorizer file not found: {self.vectorizer_path}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove emojis and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in self.stop_words
        ]
        
        # Join tokens back to text
        text = ' '.join(tokens)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_features(self, text: str, original_text: str) -> np.ndarray:
        """
        Extract features from text
        
        Args:
            text: Cleaned text
            original_text: Original text
            
        Returns:
            Feature vector
        """
        # TF-IDF features
        tfidf_features = self.vectorizer.transform([text])
        
        # Additional features
        text_length = len(text)
        word_count = len(text.split())
        exclamation_count = original_text.count('!')
        question_count = original_text.count('?')
        uppercase_ratio = sum(1 for c in original_text if c.isupper()) / (len(original_text) + 1)
        
        additional_features = np.array([[
            text_length,
            word_count,
            exclamation_count,
            question_count,
            uppercase_ratio
        ]])
        
        # Combine features
        from scipy.sparse import hstack
        combined_features = hstack([tfidf_features, additional_features])
        
        return combined_features
    
    def predict(self, text: str) -> dict:
        """
        Predict sentiment for a single text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prediction results
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text:
            return {
                'text': text,
                'sentiment': 'Unknown',
                'sentiment_label': 1,
                'confidence': 0.0,
                'probabilities': {
                    'Negative': 0.33,
                    'Neutral': 0.34,
                    'Positive': 0.33
                }
            }
        
        # Extract features
        features = self.extract_features(cleaned_text, text)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Prepare result
        result = {
            'text': text,
            'cleaned_text': cleaned_text,
            'sentiment': SENTIMENT_LABELS[prediction],
            'sentiment_label': int(prediction),
            'confidence': float(max(probabilities)),
            'probabilities': {
                SENTIMENT_LABELS[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
        }
        
        return result
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction results
        """
        return [self.predict(text) for text in texts]


if __name__ == "__main__":
    # Test the predictor
    predictor = SentimentPredictor()
    
    # Test samples
    test_comments = [
        "This is amazing! Best video ever!",
        "This is terrible. Waste of time.",
        "Okay, I guess.",
    ]
    
    logger.info("Testing predictor with sample comments...")
    for comment in test_comments:
        result = predictor.predict(comment)
        logger.info(f"\nComment: {result['text']}")
        logger.info(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f})")
        logger.info(f"Probabilities: {result['probabilities']}")
