#!/bin/bash

# Quick Start Script for API Schema Extraction - Milestone 1

set -e

echo "================================================"
echo "API Schema Extraction - Milestone 1 Quick Start"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "Error: Python 3 not found"; exit 1; }
echo "✓ Python 3 found"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Dependencies installed"
echo ""

# Check for .env file
echo "Checking configuration..."
if [ ! -f ".env" ]; then
    echo "⚠ Warning: .env file not found"
    echo ""
    echo "Please create a .env file with your GitHub token:"
    echo "  echo 'GITHUB_TOKEN=your_token_here' > .env"
    echo ""
    echo "Get a token at: https://github.com/settings/tokens"
    echo ""
else
    echo "✓ Configuration file found"
fi
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/intermediate data/final logs
echo "✓ Directories ready"
echo ""

echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. If you haven't already, add your GitHub token to .env"
echo "  2. Run the pipeline:"
echo "     python main.py --mode full --target 12"
echo ""
echo "For more information, see README.md and SETUP.md"
echo ""

