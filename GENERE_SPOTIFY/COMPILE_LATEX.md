# LaTeX Report Compilation Guide

## 📄 Document Information

**File:** `FINAL_REPORT.tex`  
**Title:** Unsupervised Music Genre Discovery Using Audio Feature Learning  
**Pages:** ~30 pages (estimated)  
**Images:** 5 PNG files from `results/` folder

---

## 🚀 Quick Compile (Recommended)

### Using Overleaf (Easiest - Online)

1. Go to [Overleaf](https://www.overleaf.com/)
2. Create a new project → Upload Project
3. Upload `FINAL_REPORT.tex`
4. Create a `results/` folder in the project
5. Upload all 5 PNG images from your local `results/` folder:
   - `feature_distributions.png`
   - `box_plots.png`
   - `correlation_heatmap.png`
   - `clustering_comparison.png`
   - `train_test_experiments.png`
6. Click "Recompile" button
7. Download PDF

---

## 💻 Local Compilation

### Prerequisites

You need a LaTeX distribution installed:

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install texlive-full
```

**macOS:**
```bash
brew install --cask mactex
```

**Windows:**
- Download and install [MiKTeX](https://miktex.org/download) or [TeX Live](https://www.tug.org/texlive/)

### Compilation Steps

#### Method 1: Using `pdflatex` (Terminal)

```bash
cd /home/anirudh-sharma/Desktop/Music\ Genere/GENERE_SPOTIFY/

# Compile three times (for references, TOC, and cross-refs)
pdflatex FINAL_REPORT.tex
pdflatex FINAL_REPORT.tex
pdflatex FINAL_REPORT.tex
```

**Output:** `FINAL_REPORT.pdf`

#### Method 2: Using `latexmk` (Automated)

```bash
cd /home/anirudh-sharma/Desktop/Music\ Genere/GENERE_SPOTIFY/

# Automatically handles all passes
latexmk -pdf FINAL_REPORT.tex
```

#### Method 3: Using TeXShop/TeXworks (GUI)

1. Open `FINAL_REPORT.tex` in TeXShop or TeXworks
2. Select "pdfLaTeX" from the typeset menu
3. Click "Typeset" button
4. Repeat 2-3 times for complete compilation

---

## 🖼️ Image Requirements

**Ensure these files exist in `results/` folder:**

```
GENERE_SPOTIFY/
├── FINAL_REPORT.tex
└── results/
    ├── box_plots.png
    ├── clustering_comparison.png
    ├── correlation_heatmap.png
    ├── feature_distributions.png
    └── train_test_experiments.png
```

**Check if images exist:**
```bash
ls -lh results/*.png
```

---

## 🐛 Troubleshooting

### Issue 1: Missing Images Error
```
LaTeX Error: File `results/feature_distributions.png' not found
```

**Solution:**
- Verify all 5 PNG files are in the `results/` folder
- Check file names match exactly (case-sensitive on Linux)
- Ensure you're compiling from the project root directory

### Issue 2: Missing Packages
```
LaTeX Error: File `booktabs.sty' not found
```

**Solution:**
```bash
# Install missing packages (Ubuntu/Debian)
sudo apt-get install texlive-latex-extra texlive-fonts-extra

# Or use MiKTeX Package Manager (Windows)
# It will auto-install missing packages
```

### Issue 3: Long Compilation Time
**Normal behavior:** LaTeX compilation can take 30-60 seconds due to images.

**Speed up:**
```bash
# Use draft mode for faster preview (no images)
pdflatex "\def\isdraft{1}\input{FINAL_REPORT.tex}"
```

### Issue 4: Bibliography Not Showing
**Solution:** This document uses manual references, not BibTeX. No action needed.

---

## 📝 Document Structure

```
FINAL_REPORT.pdf
├── Title Page
├── Table of Contents
├── Executive Summary
├── Introduction (Background, Objectives, Dataset)
├── Methodology (Preprocessing, EDA, Clustering)
├── Results (Performance, Visualizations)
├── Discussion (Algorithm Analysis, Insights)
├── Conclusions (Findings, Applications, Limitations)
├── References
└── Appendices (Technical Specs, Data Schema)
```

---

## 🎨 Customization Options

### Change Colors
Edit these lines in the preamble:
```latex
\definecolor{headerblue}{RGB}{0,51,102}  % Section headers
```

### Adjust Margins
```latex
\geometry{margin=1in}  % Change to 0.75in for narrower margins
```

### Font Size
```latex
\documentclass[12pt,a4paper]{article}  % Change to 11pt or 10pt
```

---

## 📤 Output Options

### Generate PDF/A (Archive Format)
```bash
pdflatex -output-format=pdf "\pdfminorversion=5\pdfobjcompresslevel=0\input{FINAL_REPORT.tex}"
```

### Compress PDF (Smaller Size)
```bash
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
   -dNOPAUSE -dQUIET -dBATCH \
   -sOutputFile=FINAL_REPORT_compressed.pdf FINAL_REPORT.pdf
```

---

## ✅ Verification Checklist

Before submitting, ensure:

- [ ] All 5 images display correctly in the PDF
- [ ] Table of Contents has correct page numbers
- [ ] All tables render properly
- [ ] Cross-references work (Figure~X, Table~Y)
- [ ] No overfull/underfull hbox warnings (or minimal)
- [ ] PDF file size is reasonable (<10 MB)
- [ ] All pages are correctly numbered
- [ ] Header/footer appear on all pages

---

## 🆘 Quick Help

**Can't compile locally?** → Use Overleaf (recommended)  
**Missing images?** → Check `results/` folder exists  
**PDF too large?** → Use compression command above  
**Need to edit?** → Modify `.tex` file and recompile

---

## 📚 Additional Resources

- [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX)
- [Overleaf Documentation](https://www.overleaf.com/learn)
- [TeX Stack Exchange](https://tex.stackexchange.com/)

---

**Document Created:** November 2025  
**LaTeX Version:** pdfTeX 3.141592653  
**Estimated Compilation Time:** 30-60 seconds
