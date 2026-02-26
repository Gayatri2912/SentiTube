# SentiTube - Quick Start Script
# This script sets up the environment and runs the initial pipeline

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SentiTube - Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}
Write-Host "✓ $pythonVersion" -ForegroundColor Green
Write-Host ""

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists" -ForegroundColor Yellow
} else {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}
Write-Host ""

# Activate virtual environment and install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
pip install --upgrade pip > $null 2>&1
pip install -r requirements.txt
Write-Host "✓ Dependencies installed" -ForegroundColor Green
Write-Host ""

# Download NLTK data
Write-Host "Downloading NLTK data..." -ForegroundColor Yellow
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"
Write-Host "✓ NLTK data downloaded" -ForegroundColor Green
Write-Host ""

# Initialize DVC
Write-Host "Initializing DVC..." -ForegroundColor Yellow
if (Test-Path ".dvc") {
    Write-Host "DVC already initialized" -ForegroundColor Yellow
} else {
    dvc init
    Write-Host "✓ DVC initialized" -ForegroundColor Green
}
Write-Host ""

# Initialize Git (if not already)
Write-Host "Checking Git repository..." -ForegroundColor Yellow
if (Test-Path ".git") {
    Write-Host "Git already initialized" -ForegroundColor Yellow
} else {
    git init
    git add .
    git commit -m "Initial commit - YouTube Sentiment Analysis MLOps Project"
    Write-Host "✓ Git repository initialized" -ForegroundColor Green
}
Write-Host ""

# Run the pipeline
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running ML Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will take a few minutes..." -ForegroundColor Yellow
Write-Host ""

python run_pipeline.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🎬 Welcome to SentiTube!" -ForegroundColor Yellow
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. View MLflow experiments:" -ForegroundColor White
Write-Host "   mlflow ui" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Start the web application:" -ForegroundColor White
Write-Host "   python app/app.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Or run with Docker:" -ForegroundColor White
Write-Host "   docker-compose up --build" -ForegroundColor Cyan
Write-Host ""
Write-Host "4. View documentation:" -ForegroundColor White
Write-Host "   - README.md" -ForegroundColor Cyan
Write-Host "   - GETTING_STARTED.md" -ForegroundColor Cyan
Write-Host "   - API_DOCS.md" -ForegroundColor Cyan
Write-Host ""
