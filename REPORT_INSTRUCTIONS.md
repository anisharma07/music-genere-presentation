# Comprehensive Final Report - Instructions

## Overview

This directory contains a comprehensive LaTeX report combining results from all four music genre analysis datasets:
- **GTZAN** (1,000 tracks)
- **FMA** (6,410 tracks)
- **Million Song Dataset** (100 tracks)
- **Spotify** (116,724 tracks)

## Files

- `COMPREHENSIVE_FINAL_REPORT.tex` - Main LaTeX source file
- `compile_report.sh` - Automated compilation script
- `REPORT_INSTRUCTIONS.md` - This file

## Prerequisites

### Install LaTeX

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install texlive-full
```

**macOS:**
```bash
brew install --cask mactex
```

**Windows:**
Download and install MiKTeX from https://miktex.org/download

### Verify Installation

```bash
pdflatex --version
bibtex --version
```

## Compilation Methods

### Method 1: Using the Automated Script (Recommended)

```bash
# Make the script executable
chmod +x compile_report.sh

# Run the compilation script
./compile_report.sh
```

This script will:
1. Compile the LaTeX document (first pass)
2. Process bibliography with BibTeX
3. Compile again (second pass) to resolve references
4. Final compilation for table of contents and page numbers
5. Clean up auxiliary files
6. Display success message with file size and page count

### Method 2: Manual Compilation

```bash
# First compilation
pdflatex COMPREHENSIVE_FINAL_REPORT.tex

# Process bibliography
bibtex COMPREHENSIVE_FINAL_REPORT

# Second compilation (resolve references)
pdflatex COMPREHENSIVE_FINAL_REPORT.tex

# Final compilation (finalize TOC and page numbers)
pdflatex COMPREHENSIVE_FINAL_REPORT.tex
```

### Method 3: Using LaTeX IDE

If using Overleaf, TeXShop, TeXstudio, or similar:
1. Open `COMPREHENSIVE_FINAL_REPORT.tex`
2. Click "Build" or "Compile" button
3. The IDE will handle multiple passes automatically

## Report Structure

The report follows IEEE conference format and includes:

### 1. Front Matter
- Title and author information
- Abstract (500 words)
- Keywords (5 terms)
- Table of contents

### 2. Introduction (Section I)
- Motivation
- Research objectives
- Contributions

### 3. Related Work (Section II)
- Literature review
- Previous research

### 4. Datasets (Section III)
- GTZAN overview and statistics
- FMA analysis
- Million Song Dataset details
- Spotify dataset description

### 5. Methodology (Section IV)
- Data preprocessing pipeline (4 stages)
- Statistical analysis methods
- Clustering algorithms (5 algorithms)
- Experimental configuration
- Evaluation metrics (6 metrics)

### 6. Results and Discussion (Section V)
- **GTZAN Results**
  - Statistical analysis
  - Clustering performance across splits
  - 5 visualization figures
  
- **FMA Results**
  - Statistical overview
  - Algorithm rankings
  - 6 visualization figures
  
- **MSD Results**
  - Clustering performance
  - 5 visualization figures
  
- **Spotify Results**
  - Data cleaning impact
  - Best results
  - 6 visualization figures

- **Cross-Dataset Comparison**
- **Algorithm Performance Summary**
- **Feature Importance Analysis**
- **Statistical Insights**

### 7. Discussion (Section VI)
- Performance trends
- Dataset characteristics analysis
- Algorithm strengths/weaknesses
- Practical implications
- Limitations

### 8. Conclusion (Section VII)
- Primary findings
- Contributions
- Future work (6 directions)
- Final remarks

### 9. References
- 10 academic citations

### 10. Appendices
- Dataset details
- Feature extraction parameters
- Computational environment
- Reproducibility information
- Author biography

## Embedded Visualizations

The report includes **22 figures** from all four datasets:

### GTZAN (5 figures)
1. Feature Correlation Heatmap
2. Feature Distribution Analysis
3. Outlier Detection Box Plots
4. Clustering Metrics Comparison
5. Performance Across Splits

### FMA (6 figures)
1. Feature Correlation Matrix
2. Feature Distributions
3. Box Plots
4. Accuracy Comparison
5. Silhouette Score Comparison
6. NMI Comparison

### MSD (5 figures)
1. Correlation Heatmap
2. Feature Distributions
3. Box Plots
4. t-SNE Visualization
5. Metrics Comparison

### Spotify (6 figures)
1. Feature Correlation Heatmap
2. Feature Distributions
3. Box Plots
4. Clustering Algorithm Comparison
5. Train-Test Split Experiments
6. PCA Visualization

## Tables

The report includes **10 comprehensive tables**:
1. GTZAN Statistical Summary
2. GTZAN Clustering Performance (16 experiments)
3. FMA Statistical Summary
4. FMA Algorithm Rankings
5. MSD Clustering Performance
6. Spotify Data Cleaning Summary
7. Spotify Best Results
8. Cross-Dataset Comparison
9. Overall Algorithm Rankings
10. Various appendix tables

## Expected Output

**PDF Properties:**
- **Pages:** ~25-30 pages
- **Size:** ~3-5 MB (with embedded images)
- **Format:** IEEE Conference format (two-column)
- **Quality:** Publication-ready

## Customization

### Add Your Details

Edit the following sections in the LaTeX file:

1. **Author Information** (lines 16-22):
```latex
\IEEEauthorblockN{Anirudh Sharma}
\IEEEauthorblockA{\textit{Department of Computer Science and Engineering} \\
\textit{University Name}\\  % <-- UPDATE THIS
Roll No: [Your Roll Number]\\  % <-- UPDATE THIS
Email: your.email@university.edu}  % <-- UPDATE THIS
```

2. **Computational Environment** (Appendix, Section B):
```latex
\textbf{Hardware:}
\begin{itemize}
    \item Processor: [Your processor details]  % <-- UPDATE THIS
    \item RAM: [Your RAM details]  % <-- UPDATE THIS
```

3. **Repository URL** (Appendix, Section C):
```latex
\item GitHub Repository: [Your repository URL]  % <-- UPDATE THIS
```

## Troubleshooting

### Issue: Images not found

**Solution:** Ensure the report is compiled from the root directory:
```bash
cd "/home/anirudh-sharma/Desktop/Music Genere"
./compile_report.sh
```

All image paths are relative to this directory.

### Issue: Missing packages

**Solution:** Install missing LaTeX packages:
```bash
sudo apt-get install texlive-latex-extra texlive-science
```

### Issue: Compilation errors

**Solution:** Check the log file:
```bash
cat COMPREHENSIVE_FINAL_REPORT.log | grep Error
```

### Issue: Bibliography not showing

**Solution:** Ensure you run BibTeX:
```bash
pdflatex COMPREHENSIVE_FINAL_REPORT.tex
bibtex COMPREHENSIVE_FINAL_REPORT
pdflatex COMPREHENSIVE_FINAL_REPORT.tex
pdflatex COMPREHENSIVE_FINAL_REPORT.tex
```

### Issue: Figures appear in wrong locations

**Solution:** This is normal LaTeX behavior. Use `[H]` option (already included) or let LaTeX optimize placement.

## Quality Checks

Before submission, verify:

- [ ] All figures are visible and high-quality
- [ ] All tables are properly formatted
- [ ] References are numbered correctly
- [ ] Table of contents is complete
- [ ] Your name and details are updated
- [ ] No compilation errors or warnings
- [ ] Page count is appropriate (~25-30 pages)
- [ ] PDF opens correctly

## Submission

The final PDF (`COMPREHENSIVE_FINAL_REPORT.pdf`) can be submitted as:
- Course project report
- Conference paper submission
- Technical documentation

## Additional Notes

### IEEE Format Compliance

The report uses the official `IEEEtran` document class and follows:
- Two-column layout
- IEEE citation style
- Standard section numbering
- Professional formatting

### LaTeX Packages Used

- `graphicx` - Image inclusion
- `booktabs` - Professional tables
- `multirow` - Multi-row table cells
- `hyperref` - Clickable links and references
- `subcaption` - Subfigures
- `float` - Figure placement control
- `longtable` - Multi-page tables

### Font and Spacing

- Font: Times (IEEE standard)
- Column separation: 0.25 in
- Margins: IEEE standard
- Line spacing: Single

## Contact

For questions or issues:
- Check the LaTeX log file first
- Review this instruction document
- Consult LaTeX documentation: https://www.latex-project.org/

## License

This report template and content are part of an academic project. All datasets used are publicly available with their respective licenses:
- GTZAN: Academic use
- FMA: Creative Commons
- Million Song Dataset: Research use
- Spotify: API data usage

---

**Last Updated:** November 7, 2025  
**Version:** 1.0  
**Author:** Anirudh Sharma
