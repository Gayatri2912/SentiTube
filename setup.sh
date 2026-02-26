#!/bin/bash
# SentiTube - Quick Start Script
# This script sets up the environment and runs the initial pipeline

echo "========================================"
echo "SentiTube - Setup"
echo "========================================"
echo ""

# Check if Python is installed
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi
python_version=$(python3 --version)
echo "✓ $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists"
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate virtual environment and install dependencies
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Download NLTK data
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"
echo "✓ NLTK data downloaded"
echo ""

# Initialize DVC
echo "Initializing DVC..."
if [ -d ".dvc" ]; then
    echo "DVC already initialized"
else
    dvc init
    echo "✓ DVC initialized"
fi
echo ""

# Initialize Git (if not already)
echo "Checking Git repository..."
if [ -d ".git" ]; then
    echo "Git already initialized"
else
    git init
    git add .
    git commit -m "Initial commit - YouTube Sentiment Analysis MLOps Project"
    echo "✓ Git repository initialized"
fi
echo ""

# Run the pipeline
echo "========================================"
echo "Running ML Pipeline"
echo "========================================"
echo ""
echo "This will take a few minutes..."
echo ""

python3 run_pipeline.py

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "🎬 Welcome to SentiTube!"
echo ""
echo "Next steps:"
echo "1. View MLflow experiments:"
echo "   mlflow ui"
echo ""
echo "2. Start the web application:"
echo "   python app/app.py"
echo ""
echo "3. Or run with Docker:"
echo "   docker-compose up --build"
echo ""
echo "4. View documentation:"
echo "   - README.md"
echo "   - GETTING_STARTED.md"
echo "   - API_DOCS.md"
echo ""
