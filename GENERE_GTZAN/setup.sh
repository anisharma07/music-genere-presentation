#!/bin/bash

# Installation and Setup Script for Music Genre Discovery Project
# Author: Anirudh Sharma
# Date: November 2025

echo "=========================================="
echo "Music Genre Discovery Project Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "Found: $python_version"

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed!"
    echo "Please install Python 3.8 or higher."
    exit 1
fi

echo ""
echo "Installing required packages..."
echo ""

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install requirements
echo ""
echo "Installing project dependencies..."
python3 -m pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Installation completed successfully!"
    echo "=========================================="
    echo ""
    echo "To run the project:"
    echo "  python3 main.py"
    echo ""
    echo "Individual modules:"
    echo "  python3 data_analysis.py           # Data analysis only"
    echo "  python3 clustering_implementation.py   # Clustering only"
    echo "  python3 cross_validation.py        # Cross-validation only"
    echo ""
else
    echo ""
    echo "ERROR: Installation failed!"
    echo "Please check your Python environment and try again."
    exit 1
fi
