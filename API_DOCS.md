# SentiTube - API Documentation

## API Endpoints

### Health Check

**Endpoint:** `GET /health`

**Description:** Check if the API is running and the model is loaded.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

---

### Single Prediction

**Endpoint:** `POST /predict`

**Description:** Predict sentiment for a single comment.

**Request Body:**
```json
{
    "text": "This is an amazing video! Love it!"
}
```

**Response:**
```json
{
    "text": "This is an amazing video! Love it!",
    "cleaned_text": "amazing video love",
    "sentiment": "Positive",
    "sentiment_label": 2,
    "confidence": 0.95,
    "probabilities": {
        "Negative": 0.02,
        "Neutral": 0.03,
        "Positive": 0.95
    }
}
```

**Status Codes:**
- `200 OK` - Successful prediction
- `400 Bad Request` - Invalid request (missing text, empty text)
- `500 Internal Server Error` - Model not loaded or prediction error

---

### Batch Prediction

**Endpoint:** `POST /predict_batch`

**Description:** Predict sentiment for multiple comments.

**Request Body:**
```json
{
    "texts": [
        "This is amazing!",
        "Terrible video.",
        "It's okay I guess."
    ]
}
```

**Response:**
```json
{
    "predictions": [
        {
            "text": "This is amazing!",
            "sentiment": "Positive",
            "sentiment_label": 2,
            "confidence": 0.92,
            "probabilities": {
                "Negative": 0.03,
                "Neutral": 0.05,
                "Positive": 0.92
            }
        },
        {
            "text": "Terrible video.",
            "sentiment": "Negative",
            "sentiment_label": 0,
            "confidence": 0.88,
            "probabilities": {
                "Negative": 0.88,
                "Neutral": 0.08,
                "Positive": 0.04
            }
        },
        {
            "text": "It's okay I guess.",
            "sentiment": "Neutral",
            "sentiment_label": 1,
            "confidence": 0.65,
            "probabilities": {
                "Negative": 0.18,
                "Neutral": 0.65,
                "Positive": 0.17
            }
        }
    ],
    "count": 3
}
```

**Status Codes:**
- `200 OK` - Successful predictions
- `400 Bad Request` - Invalid request (missing texts, not a list, empty list)
- `500 Internal Server Error` - Model not loaded or prediction error

---

## Usage Examples

### cURL

**Health Check:**
```bash
curl http://localhost:8080/health
```

**Single Prediction:**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is an amazing video!"}'
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:8080/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Amazing!", "Terrible!", "Okay."]}'
```

### Python

```python
import requests

# Base URL
API_URL = "http://localhost:8080"

# Health check
response = requests.get(f"{API_URL}/health")
print(response.json())

# Single prediction
response = requests.post(
    f"{API_URL}/predict",
    json={"text": "This is an amazing video!"}
)
print(response.json())

# Batch prediction
response = requests.post(
    f"{API_URL}/predict_batch",
    json={
        "texts": [
            "This is amazing!",
            "Terrible video.",
            "It's okay."
        ]
    }
)
print(response.json())
```

### JavaScript

```javascript
// Single prediction
fetch('http://localhost:8080/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        text: 'This is an amazing video!'
    })
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));

// Batch prediction
fetch('http://localhost:8080/predict_batch', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        texts: ['Amazing!', 'Terrible!', 'Okay.']
    })
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));
```

---

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Original input text |
| `cleaned_text` | string | Preprocessed text used for prediction |
| `sentiment` | string | Sentiment label (Positive/Neutral/Negative) |
| `sentiment_label` | integer | Numeric sentiment (0=Negative, 1=Neutral, 2=Positive) |
| `confidence` | float | Confidence score (0-1) |
| `probabilities` | object | Probability for each sentiment class |

---

## Error Responses

**Missing Text Field:**
```json
{
    "error": "Missing \"text\" field in request"
}
```

**Empty Text:**
```json
{
    "error": "Empty text provided"
}
```

**Model Not Loaded:**
```json
{
    "error": "Model not loaded"
}
```

**Invalid Request:**
```json
{
    "error": "\"texts\" must be a list"
}
```

---

## Rate Limiting

Currently, there is no rate limiting implemented. For production deployment, consider:
- Implementing rate limiting middleware
- Adding authentication/API keys
- Using a reverse proxy (nginx) with rate limiting
- Deploying behind an API gateway

---

## CORS

CORS is enabled for all origins. For production, restrict origins in `app/app.py`:

```python
from flask_cors import CORS

CORS(app, resources={
    r"/*": {
        "origins": ["https://yourdomain.com"]
    }
})
```

---

## Performance

- Single prediction: ~50-100ms
- Batch prediction: ~200-500ms for 10 comments
- Memory usage: ~500MB with model loaded

For better performance:
- Use batch predictions when possible
- Consider deploying multiple instances behind a load balancer
- Cache frequent predictions
- Use async workers (Gunicorn with gevent)

---

## Security Considerations

For production deployment:
1. Add authentication (API keys, OAuth)
2. Implement rate limiting
3. Add input validation and sanitization
4. Enable HTTPS
5. Set up monitoring and logging
6. Use environment variables for sensitive config
7. Implement request timeouts
8. Add CSRF protection if needed
