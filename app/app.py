"""
SentiTube - Flask Web Application
AI-Powered YouTube Sentiment Analysis Platform
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.prediction import SentimentPredictor
from src.config import API_HOST, API_PORT, API_DEBUG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize predictor
try:
    predictor = SentimentPredictor()
    logger.info("Sentiment predictor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize predictor: {e}")
    predictor = None


@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment for a single comment
    
    Request JSON:
        {
            "text": "Your comment here"
        }
    
    Response JSON:
        {
            "text": "Your comment here",
            "sentiment": "Positive",
            "sentiment_label": 2,
            "confidence": 0.95,
            "probabilities": {
                "Negative": 0.02,
                "Neutral": 0.03,
                "Positive": 0.95
            }
        }
    """
    if predictor is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    try:
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({
                'error': 'Missing "text" field in request'
            }), 400
        
        text = data['text']
        
        if not text or not text.strip():
            return jsonify({
                'error': 'Empty text provided'
            }), 400
        
        # Make prediction
        result = predictor.predict(text)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict sentiment for multiple comments
    
    Request JSON:
        {
            "texts": ["Comment 1", "Comment 2", ...]
        }
    
    Response JSON:
        {
            "predictions": [
                {
                    "text": "Comment 1",
                    "sentiment": "Positive",
                    ...
                },
                ...
            ],
            "count": 2
        }
    """
    if predictor is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    try:
        data = request.get_json()
        
        if 'texts' not in data:
            return jsonify({
                'error': 'Missing "texts" field in request'
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({
                'error': '"texts" must be a list'
            }), 400
        
        if len(texts) == 0:
            return jsonify({
                'error': 'Empty texts list'
            }), 400
        
        # Make predictions
        results = predictor.predict_batch(texts)
        
        return jsonify({
            'predictions': results,
            'count': len(results)
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':
    logger.info(f"Starting Flask app on {API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)
