#!/bin/bash

# Verification script for LaTeX report compilation
# Checks all dependencies and files before compilation

echo "=========================================="
echo "📋 Report Compilation Pre-Check"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# Function to check command
check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 found: $(command -v $1)"
        return 0
    else
        echo -e "${RED}✗${NC} $1 NOT FOUND"
        ((ERRORS++))
        return 1
    fi
}

# Function to check file
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} Found: $1"
        return 0
    else
        echo -e "${RED}✗${NC} Missing: $1"
        ((ERRORS++))
        return 1
    fi
}

# Function to check directory
check_dir() {
    if [ -d "$1" ]; then
        count=$(ls "$1" 2>/dev/null | wc -l)
        echo -e "${GREEN}✓${NC} Found: $1 ($count files)"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} Missing directory: $1"
        ((WARNINGS++))
        return 1
    fi
}

echo "1. Checking LaTeX Installation"
echo "----------------------------"
check_command pdflatex
check_command bibtex
echo ""

echo "2. Checking Main Files"
echo "----------------------------"
check_file "COMPREHENSIVE_FINAL_REPORT.tex"
check_file "compile_report.sh"
check_file "REPORT_INSTRUCTIONS.md"
echo ""

echo "3. Checking Dataset Directories"
echo "----------------------------"
check_dir "GENERE_GTZAN/output/results"
check_dir "GENERE_FMA/output/results"
check_dir "GENERE_MSD/output/plots"
check_dir "GENERE_SPOTIFY/output/results"
check_dir "GENERE_SPOTIFY/visualizations"
echo ""

echo "4. Checking Key Visualization Files"
echo "----------------------------"

# GTZAN
echo -e "${YELLOW}GTZAN:${NC}"
check_file "GENERE_GTZAN/output/results/correlation_heatmap.png"
check_file "GENERE_GTZAN/output/results/distribution_analysis.png"
check_file "GENERE_GTZAN/output/results/metrics_comparison.png"

# FMA
echo -e "${YELLOW}FMA:${NC}"
check_file "GENERE_FMD/output/results/correlation_matrix.png"
check_file "GENERE_FMD/output/results/distributions.png"
check_file "GENERE_FMD/output/results/accuracy_comparison.png"

# MSD
echo -e "${YELLOW}MSD:${NC}"
check_file "GENERE_MSD/output/plots/correlation_heatmap.png"
check_file "GENERE_MSD/output/plots/tsne_visualization.png"
check_file "GENERE_MSD/output/plots/metrics_comparison.png"

# Spotify
echo -e "${YELLOW}Spotify:${NC}"
check_file "GENERE_SPOTIFY/visualizations/correlation_heatmap.png"
check_file "GENERE_SPOTIFY/visualizations/feature_distributions.png"
check_file "GENERE_SPOTIFY/output/results/clustering_comparison.png"

echo ""
echo "5. Checking CSV Result Files"
echo "----------------------------"
check_file "GENERE_GTZAN/output/results/clustering_results.csv"
check_file "GENERE_MSD/output/results/evaluation_metrics.csv"
echo ""

echo "=========================================="
echo "📊 Pre-Check Summary"
echo "=========================================="

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    echo ""
    echo "Ready to compile. Run:"
    echo "  ./compile_report.sh"
else
    echo -e "${RED}✗ $ERRORS critical error(s) found${NC}"
    echo ""
    echo "Please fix the errors above before compiling."
fi

if [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}⚠ $WARNINGS warning(s) - non-critical${NC}"
fi

echo ""
echo "=========================================="
echo ""

exit $ERRORS
