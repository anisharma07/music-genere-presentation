#!/bin/bash

# Quick Start Script for Music Genre Discovery Project
# This script sets up the environment and runs the complete pipeline

echo "=========================================="
echo "Music Genre Discovery - Quick Start"
echo "=========================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✓ Dependencies installed"
echo ""

# Verify installation
echo "Verifying installation..."
python3 -c "import h5py, numpy, pandas, sklearn; print('✓ All packages verified')"
echo ""

# Check if dataset exists
if [ ! -d "million song/millionsongsubset/MillionSongSubset" ]; then
    echo "WARNING: Dataset not found at 'million song/millionsongsubset/MillionSongSubset'"
    echo "Please ensure the Million Song Dataset is in the correct location."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run the pipeline
echo "=========================================="
echo "Starting Pipeline..."
echo "=========================================="
echo ""

python3 main.py

# Check if pipeline succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Pipeline completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results are available in:"
    echo "  - output/              (all outputs)"
    echo "  - output/results/      (CSV results)"
    echo "  - output/plots/        (visualizations)"
    echo "  - output/models/       (trained models)"
    echo ""
    echo "Check output/results/final_report.txt for a summary."
else
    echo ""
    echo "=========================================="
    echo "✗ Pipeline failed"
    echo "=========================================="
    echo ""
    echo "Check output/pipeline.log for details."
    exit 1
fi
