"""Unit tests for data preprocessing module"""
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.data_preprocessing import DataPreprocessing


class TestDataPreprocessing:
    """Test cases for DataPreprocessing class"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a DataPreprocessing instance"""
        return DataPreprocessing()
    
    def test_clean_text_basic(self, preprocessor):
        """Test basic text cleaning"""
        text = "This is AMAZING! Best video ever! 😊"
        cleaned = preprocessor.clean_text(text)
        
        assert isinstance(cleaned, str)
        assert len(cleaned) > 0
        assert cleaned.islower()
    
    def test_clean_text_urls(self, preprocessor):
        """Test URL removal"""
        text = "Check this out https://example.com great video!"
        cleaned = preprocessor.clean_text(text)
        
        assert "https" not in cleaned
        assert "example.com" not in cleaned
    
    def test_clean_text_mentions(self, preprocessor):
        """Test mention removal"""
        text = "@user this is a great video!"
        cleaned = preprocessor.clean_text(text)
        
        assert "@user" not in cleaned
    
    def test_clean_text_empty(self, preprocessor):
        """Test handling of empty text"""
        text = ""
        cleaned = preprocessor.clean_text(text)
        
        assert cleaned == ""
    
    def test_clean_text_none(self, preprocessor):
        """Test handling of None"""
        text = None
        cleaned = preprocessor.clean_text(text)
        
        assert cleaned == ""
    
    def test_preprocess_data(self, preprocessor):
        """Test full preprocessing pipeline"""
        df = pd.DataFrame({
            'comment': ['Great video!', 'Bad content', 'OK'],
            'sentiment': [2, 0, 1]
        })
        
        df_processed = preprocessor.preprocess_data(df)
        
        assert 'cleaned_comment' in df_processed.columns
        assert len(df_processed) <= len(df)  # Some might be filtered
        assert all(df_processed['cleaned_comment'].str.len() >= preprocessor.params['min_text_length'])


if __name__ == "__main__":
    pytest.main([__file__])
