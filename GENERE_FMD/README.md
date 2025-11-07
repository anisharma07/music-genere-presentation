# Unsupervised Music Genre Discovery Using Audio Feature Learning

A comprehensive implementation of unsupervised music genre classification using the FMA (Free Music Archive) dataset with multiple clustering algorithms and evaluation metrics.

## 📋 Project Overview

This project implements an end-to-end pipeline for discovering music genres through audio feature analysis and unsupervised clustering algorithms. It includes:

- **Feature Extraction**: MFCCs, Chroma, Tempo, Spectral features
- **Data Analysis**: Statistical analysis, outlier detection, correlation analysis
- **Clustering Algorithms**: K-Means, Spectral Clustering, DBSCAN, GMM
- **Evaluation Metrics**: 8+ metrics including Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, NMI, and more
- **Multiple Experiments**: Train-test splits of 50-50, 60-40, 70-30, and 80-20

## 🗂️ Project Structure

```
GENERE_FMD/
├── fma_small/              # FMA dataset directory (MP3 files)
├── requirements.txt        # Python dependencies
├── feature_extraction.py   # Audio feature extraction module
├── data_analysis.py        # Data analysis and cleaning module
├── clustering.py           # Clustering algorithms implementation
├── evaluation.py           # Evaluation metrics module
├── main.py                 # Main pipeline orchestrator
├── download_metadata.py    # Helper script to download FMA metadata
├── README.md              # This file
└── results/               # Output directory (created automatically)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Download FMA Metadata (Optional)

If you want to use actual genre labels instead of synthetic labels:

```bash
python download_metadata.py
```

This will download the FMA metadata CSV files containing track information and genre labels.

### 3. Run the Complete Pipeline

```bash
python main.py
```

The pipeline will:
1. Extract audio features from MP3 files
2. Perform comprehensive data analysis
3. Clean and preprocess the data
4. Run clustering experiments with multiple algorithms
5. Evaluate using 8+ metrics
6. Generate comparison tables and visualizations

## 📊 Features Extracted

### Audio Features (67 total)
- **MFCCs**: 20 coefficients (mean + std) = 40 features
- **Chroma**: 12 pitch classes (mean + std) = 24 features
- **Spectral Features**: Centroid, Rolloff, Bandwidth (mean + std) = 6 features
- **Other**: Zero Crossing Rate, RMS Energy, Tempo = 3 features

### Dimensionality Reduction
- PCA is applied to reduce features to 20 dimensions
- Preserves ~85-95% of variance

## 🔬 Algorithms Implemented

### Clustering Algorithms
1. **K-Means**: Standard clustering with K=10
2. **MiniBatch K-Means**: Scalable variant for large datasets
3. **Spectral Clustering**: Graph-based clustering
4. **DBSCAN**: Density-based clustering with automatic cluster detection
5. **GMM**: Probabilistic Gaussian Mixture Model

### Evaluation Metrics

**Internal Metrics** (no ground truth needed):
- Silhouette Score (higher is better)
- Davies-Bouldin Index (lower is better)
- Calinski-Harabasz Index (higher is better)

**External Metrics** (with ground truth):
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- V-Measure Score
- Purity Index
- Cluster Accuracy

## 📈 Data Analysis Pipeline

### Statistical Analysis
- Descriptive statistics (mean, std, min, max, quartiles)
- Skewness and kurtosis analysis
- Percentile calculations (25th, 75th)
- Distribution pattern identification

### Data Cleaning
1. **Missing Values**: Filled with column mean
2. **Outlier Detection**: IQR method (1.5 × IQR)
3. **Outlier Removal**: Configurable threshold
4. **Trimmed Statistics**: 10% trimming from each end

### Visualizations
- Box plots for outlier detection
- Distribution histograms
- Correlation heatmaps
- Metric comparison charts

## 🎯 Experimental Setup

### Train-Test Splits
- 50-50 (50% train, 50% test)
- 60-40 (60% train, 40% test)
- 70-30 (70% train, 30% test)
- 80-20 (80% train, 20% test)

### Cross-Validation
- Random state fixed for reproducibility
- Multiple runs with different splits

## 📝 Output Files

The pipeline generates the following files in the `results/` directory:

### Data Files
- `extracted_features.csv`: Raw audio features
- `cleaned_features.csv`: Processed features after cleaning
- `comparison_table_*.csv`: Evaluation metrics for each split

### Visualizations
- `boxplots.png`: Outlier visualization
- `distributions.png`: Feature distributions
- `correlation_matrix.png`: Feature correlations
- `silhouette_comparison.png`: Silhouette score comparison
- `davies-bouldin_comparison.png`: DBI comparison
- `calinski-harabasz_comparison.png`: CHI comparison
- `comprehensive_heatmap_*.png`: All metrics heatmap

## 🔧 Configuration

Edit `main.py` to customize:

```python
DATA_PATH = 'fma_small'      # Dataset path
OUTPUT_DIR = 'results'        # Output directory
MAX_FILES = 500              # Number of files to process (None = all)
N_CLUSTERS = 10              # Number of clusters
```

## 📊 Sample Results Table

| Algorithm | #Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz | NMI | ARI | V-Measure | Accuracy |
|-----------|-----------|------------|----------------|-------------------|-----|-----|-----------|----------|
| K-Means | 10 | 0.41 | 0.86 | 240 | 0.52 | 0.47 | 0.55 | 0.60 |
| Spectral Clustering | 10 | 0.57 | 0.52 | 310 | 0.68 | 0.63 | 0.70 | 0.73 |
| DBSCAN | auto | 0.45 | 0.70 | 280 | 0.60 | 0.55 | 0.61 | 0.66 |
| GMM | 10 | 0.50 | 0.65 | 295 | 0.63 | 0.58 | 0.66 | 0.71 |

*Note: Actual results will vary based on the data*

## 🎓 Academic Use

This project fulfills the requirements for:
- Data adequacy analysis
- Descriptive statistical analysis
- Outlier detection and removal
- Missing value handling
- Distribution pattern identification
- Correlation analysis
- Multiple clustering algorithms
- Comprehensive evaluation metrics
- Train-test split experiments

## 📚 References

- FMA Dataset: https://github.com/mdeff/fma
- Paper: https://arxiv.org/abs/1612.01840
- Librosa Documentation: https://librosa.org/
- Scikit-learn Clustering: https://scikit-learn.org/stable/modules/clustering.html

## 🐛 Troubleshooting

### Memory Issues
If you encounter memory errors:
- Reduce `MAX_FILES` in `main.py`
- Use MiniBatch K-Means instead of standard K-Means
- Reduce PCA components

### Slow Processing
- Reduce audio duration in `AudioFeatureExtractor` (default: 30s)
- Use smaller batch sizes
- Process files in chunks

### No Audio Files Found
Ensure the `fma_small` directory contains MP3 files in subdirectories (000/, 001/, etc.)

## 📄 License

This project uses the FMA dataset. Each MP3 file is licensed by its artist.

## 👥 Contributors

- Project for: Unsupervised Music Genre Discovery
- Dataset: Free Music Archive (FMA)

## 🔄 Future Improvements

- [ ] Add support for FMA medium/large datasets
- [ ] Implement deep learning features (CNN-based)
- [ ] Add real-time genre prediction
- [ ] Create interactive visualizations
- [ ] Implement ensemble clustering methods
