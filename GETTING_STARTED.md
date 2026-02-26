# SentiTube - Getting Started Guide

## 🚀 Quick Start Guide

### Step 1: Environment Setup

1. **Clone or navigate to the project directory:**
```bash
cd youtube
```

2. **Create and activate a virtual environment:**

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Step 2: Run the Complete Pipeline

**Option 1: Run the entire pipeline at once**
```bash
python run_pipeline.py
```

This will execute all steps:
- Data ingestion
- Data preprocessing
- Feature engineering
- Model training (Random Forest, XGBoost, LightGBM)
- Model evaluation

**Option 2: Run steps individually**

```bash
# Step 1: Data ingestion
python src/data_ingestion.py

# Step 2: Preprocessing
python src/data_preprocessing.py

# Step 3: Feature engineering
python src/feature_engineering.py

# Step 4: Train models
python src/model_training.py

# Step 5: Evaluate models
python src/model_evaluation.py
```

### Step 3: View Experiments in MLflow

After training, view experiment results:

```bash
mlflow ui
```

Then open your browser to: http://localhost:5000

### Step 4: Run the Web Application

Start the Flask API:

```bash
python app/app.py
```

Access the web interface at: http://localhost:8080

### Step 5: Make Predictions

**Via Web Interface:**
- Open http://localhost:8080
- Enter a YouTube comment
- Click "Analyze Sentiment"

**Via API (cURL):**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is an amazing video!"}'
```

**Via Python:**
```python
import requests

response = requests.post(
    'http://localhost:8080/predict',
    json={'text': 'This is an amazing video!'}
)

print(response.json())
```

### Step 6: Run with Docker

**Build and run with Docker Compose:**
```bash
docker-compose up --build
```

This starts:
- SentiTube API on port 8080
- MLflow tracking server on port 5000

**Access:**
- API: http://localhost:8080
- MLflow: http://localhost:5000

### Step 7: Explore the Data (Optional)

Run the EDA notebook:

```bash
jupyter notebook notebooks/EDA.ipynb
```

## 🧪 Testing

Run unit tests:

```bash
pytest tests/ -v
```

Run tests with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## 📊 Project Components

### Data Flow
```
Raw Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment
```

### Models Trained
1. **Random Forest** - Ensemble baseline
2. **XGBoost** - Gradient boosting with regularization
3. **LightGBM** - Fast gradient boosting (typically best performer)

### Key Features
- **TF-IDF Vectorization** (5000 features, bigrams)
- **Text length**
- **Word count**
- **Exclamation count**
- **Question mark count**
- **Uppercase ratio**

### Handling Imbalanced Data
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- Applied to training data only
- Configurable in `params.yaml`

## 🔧 Configuration

Modify `params.yaml` to adjust:
- Model hyperparameters
- Feature engineering settings
- Data preprocessing options
- SMOTE parameters
- API configuration

## 📁 Project Structure

```
youtube/
├── src/                    # Source code
├── app/                    # Flask web application
├── data/                   # Data directory
├── models/                 # Saved models
├── mlruns/                # MLflow experiments
├── metrics/               # Evaluation metrics
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── .github/workflows/     # CI/CD pipelines
├── requirements.txt       # Dependencies
├── params.yaml           # Configuration
├── dvc.yaml              # DVC pipeline
└── run_pipeline.py       # Pipeline runner
```

## 🐛 Troubleshooting

### Issue: NLTK data not found
**Solution:**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Issue: Model file not found
**Solution:** Run the pipeline first:
```bash
python run_pipeline.py
```

### Issue: Port already in use
**Solution:** Change port in `params.yaml` or kill the process:

**Windows:**
```powershell
netstat -ano | findstr :8080
taskkill /PID <PID> /F
```

**Linux/Mac:**
```bash
lsof -ti:8080 | xargs kill -9
```

## 📚 Next Steps

1. **Customize the dataset**: Replace synthetic data with real YouTube comments
2. **Tune hyperparameters**: Modify `params.yaml` and retrain
3. **Add more features**: Enhance `feature_engineering.py`
4. **Deploy to cloud**: Use the CI/CD pipeline to deploy to AWS/Azure/GCP
5. **Monitor production**: Set up model monitoring and drift detection

## 📖 Documentation

- [README.md](README.md) - Project overview
- [GETTING_STARTED.md](GETTING_STARTED.md) - This file
- [params.yaml](params.yaml) - Configuration reference
- [notebooks/EDA.ipynb](notebooks/EDA.ipynb) - Data exploration

## 💡 Tips

- Use `mlflow ui` to compare experiment results
- Adjust SMOTE parameters if dealing with severe imbalance
- Monitor training time vs. performance trade-offs
- Use Docker for consistent deployment environments
- Leverage DVC for data versioning in production

## 🤝 Contributing

Feel free to:
- Add new models
- Improve preprocessing
- Enhance the UI
- Add more test cases
- Improve documentation

Happy coding! 🎉
