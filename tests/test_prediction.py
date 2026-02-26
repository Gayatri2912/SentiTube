"""Unit tests for prediction module"""
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


class TestPrediction:
    """Test cases for SentimentPredictor class"""
    
    def test_import(self):
        """Test that prediction module can be imported"""
        try:
            from src.prediction import SentimentPredictor
            assert SentimentPredictor is not None
        except ImportError:
            pytest.skip("Prediction module requires trained model")
    
    @pytest.mark.skipif(
        not Path('../models/best_model.pkl').exists(),
        reason="Model not trained yet"
    )
    def test_predictor_initialization(self):
        """Test predictor initialization"""
        from src.prediction import SentimentPredictor
        
        predictor = SentimentPredictor()
        assert predictor.model is not None
        assert predictor.vectorizer is not None
    
    @pytest.mark.skipif(
        not Path('../models/best_model.pkl').exists(),
        reason="Model not trained yet"
    )
    def test_predict_positive(self):
        """Test prediction on positive comment"""
        from src.prediction import SentimentPredictor
        
        predictor = SentimentPredictor()
        result = predictor.predict("This is amazing! Best video ever!")
        
        assert 'sentiment' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert 0 <= result['confidence'] <= 1
    
    @pytest.mark.skipif(
        not Path('../models/best_model.pkl').exists(),
        reason="Model not trained yet"
    )
    def test_predict_batch(self):
        """Test batch prediction"""
        from src.prediction import SentimentPredictor
        
        predictor = SentimentPredictor()
        texts = [
            "Great video!",
            "Terrible content.",
            "It's okay."
        ]
        
        results = predictor.predict_batch(texts)
        
        assert len(results) == len(texts)
        assert all('sentiment' in r for r in results)


if __name__ == "__main__":
    pytest.main([__file__])
