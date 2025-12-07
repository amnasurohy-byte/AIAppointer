#!/bin/bash
# Installation script for AI Appointer Assist (Air-Gapped Environment)

set -e

echo "========================================="
echo "AI Appointer Assist - Installation"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "Error: Python 3.8+ required. Found: $python_version"
    exit 1
fi
echo "✓ Python $python_version detected"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies from local wheels (air-gapped)
if [ -d "wheels" ]; then
    echo "Installing dependencies from local wheels..."
    pip install --no-index --find-links=wheels/ -r requirements.txt
    echo "✓ Dependencies installed from wheels"
else
    echo "Installing dependencies from PyPI..."
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
fi
echo ""

# Validate installation
echo "Validating installation..."
python3 -c "import streamlit, pandas, numpy, sklearn, lightgbm, joblib; print('✓ All core dependencies OK')"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p data models logs
echo "✓ Directories created"
echo ""

# Set permissions
echo "Setting permissions..."
chmod +x scripts/*.sh 2>/dev/null || true
echo "✓ Permissions set"
echo ""

echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Place your dataset in: data/"
echo "2. Place trained models in: models/"
echo "3. Configure: cp .env.example .env && nano .env"
echo "4. Start application: ./scripts/start.sh"
echo ""
