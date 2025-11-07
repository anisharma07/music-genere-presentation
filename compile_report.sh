#!/bin/bash

# Comprehensive Final Report Compilation Script
# This script compiles the LaTeX report with all references and figures

echo "=========================================="
echo "Compiling Comprehensive Final Report"
echo "=========================================="

# Check if pdflatex is installed
if ! command -v pdflatex &> /dev/null
then
    echo "ERROR: pdflatex not found. Please install texlive:"
    echo "  sudo apt-get install texlive-full"
    exit 1
fi

# Define the main tex file
MAIN_FILE="COMPREHENSIVE_FINAL_REPORT"

echo ""
echo "Step 1: First LaTeX compilation..."
pdflatex -interaction=nonstopmode "$MAIN_FILE.tex"

echo ""
echo "Step 2: BibTeX compilation..."
bibtex "$MAIN_FILE"

echo ""
echo "Step 3: Second LaTeX compilation..."
pdflatex -interaction=nonstopmode "$MAIN_FILE.tex"

echo ""
echo "Step 4: Final LaTeX compilation..."
pdflatex -interaction=nonstopmode "$MAIN_FILE.tex"

# Clean up auxiliary files
echo ""
echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.lof *.lot

echo ""
echo "=========================================="
if [ -f "$MAIN_FILE.pdf" ]; then
    echo "✅ SUCCESS! PDF generated: $MAIN_FILE.pdf"
    echo "=========================================="
    echo ""
    echo "File size: $(du -h "$MAIN_FILE.pdf" | cut -f1)"
    echo "Pages: $(pdfinfo "$MAIN_FILE.pdf" 2>/dev/null | grep Pages | awk '{print $2}')"
else
    echo "❌ ERROR: PDF generation failed"
    echo "=========================================="
    echo "Check the LaTeX log file for errors"
fi

echo ""
