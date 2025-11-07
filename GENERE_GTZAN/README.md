# Unsupervised Music Genre Discovery Using Audio Feature Learning

## Project Overview

This project implements an unsupervised machine learning system for discovering music genres using audio features from the GTZAN dataset. The system uses multiple clustering algorithms and comprehensive evaluation metrics to identify patterns in music data without labeled supervision.

**Author:** Anirudh Sharma  
**Topic:** Unsupervised Music Genre Discovery Using Audio Feature Learning  
**Date:** November 2025

## Project Structure

```
GENERE_GTZAN/
├── main.py                          # Main execution script
├── data_analysis.py                 # Data analysis and cleaning module
├── clustering_implementation.py     # Clustering algorithms implementation
├── cross_validation.py              # Cross-validation module
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── TO_DO.md                        # Project requirements and tasks
├── gtzan/                          # Dataset directory
│   ├── features_30_sec.csv         # 30-second audio features
│   ├── features_3_sec.csv          # 3-second audio features
│   ├── gtzan_metadata.csv          # Metadata
│   └── genres/                     # Audio files by genre
├── results/                        # Generated results (created on run)
└── visualizations/                 # Generated plots (created on run)
```

## Features

### Data Analysis Module (`data_analysis.py`)
- **Data Adequacy Check**: Validates dataset size and quality
- **Class Balance Analysis**: Checks genre distribution
- **Descriptive Statistics**: Comprehensive statistical analysis
- **Outlier Detection**: IQR and Z-score methods with boxplots
- **Missing Value Handling**: Identifies and fills missing data
- **Distribution Analysis**: Tests for normality and visualizes distributions
- **Percentile & Quartile Analysis**: Calculates key statistics
- **Trimmed Statistics**: Robust statistics with outlier trimming
- **Correlation Analysis**: Pearson correlation with heatmaps

### Clustering Module (`clustering_implementation.py`)
Implements 4 clustering algorithms:
1. **K-Means / MiniBatch K-Means**: Centroid-based clustering
2. **Spectral Clustering**: Graph-based clustering
3. **DBSCAN**: Density-based clustering
4. **Gaussian Mixture Model (GMM)**: Probabilistic clustering

### Evaluation Metrics

#### Internal Metrics (Unsupervised)
- **Silhouette Score**: Measures cluster cohesion and separation
- **Davies-Bouldin Index**: Evaluates cluster separation (lower is better)
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance

#### External Metrics (Supervised - using ground truth)
- **Adjusted Rand Index (ARI)**: Similarity to ground truth
- **Normalized Mutual Information (NMI)**: Information theoretic measure
- **V-Measure**: Harmonic mean of homogeneity and completeness
- **Cluster Accuracy**: Best alignment with true labels

### Cross-Validation Module (`cross_validation.py`)
- 5-fold stratified cross-validation
- Robust performance estimation
- Statistical significance testing

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory:**
```bash
cd "/home/anirudh-sharma/Desktop/Music Genere/GENERE_GTZAN"
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

## Usage

### Quick Start

Run the complete pipeline:
```bash
python main.py
```

This will:
1. Perform comprehensive data analysis
2. Clean and preprocess the dataset
3. Run clustering experiments with multiple train-test splits (50-50, 60-40, 70-30, 80-20)
4. Evaluate using all metrics
5. Generate visualizations and reports
6. Save all results in the `results/` directory

### Individual Modules

**Data Analysis Only:**
```bash
python data_analysis.py
```

**Clustering Experiments Only:**
```bash
python clustering_implementation.py
```

**Cross-Validation:**
```bash
python cross_validation.py
```

## Generated Outputs

### Data Analysis Results
- `results/class_balance.png` - Genre distribution visualization
- `results/descriptive_statistics.csv` - Complete statistical summary
- `results/outlier_boxplots.png` - Outlier detection visualizations
- `results/distribution_analysis.png` - Feature distributions
- `results/correlation_heatmap.png` - Feature correlation matrix
- `gtzan/features_30_sec_cleaned.csv` - Cleaned dataset

### Clustering Results
- `results/clustering_results.csv` - Detailed metrics for all experiments
- `results/summary_table.csv` - Average performance across splits
- `results/metrics_comparison.png` - Algorithm comparison charts
- `results/performance_by_split.png` - Performance trends
- `results/radar_chart.png` - Multi-metric algorithm comparison
- `results/cluster_viz_*.png` - 2D PCA visualizations for each algorithm

### Cross-Validation Results
- `results/cross_validation_results.csv` - Fold-wise results
- `results/cross_validation_summary.csv` - Mean and std statistics
- `results/cross_validation_boxplots.png` - Performance distribution

## Workflow

The project follows this workflow:

```
1. Load Dataset (GTZAN features)
   ↓
2. Data Analysis & Cleaning
   - Check adequacy and balance
   - Detect and remove outliers
   - Handle missing values
   - Statistical analysis
   ↓
3. Feature Preprocessing
   - Standardization (Z-score normalization)
   - PCA dimensionality reduction (→ 20 components)
   ↓
4. Clustering Experiments
   - Split data (50-50, 60-40, 70-30, 80-20)
   - Apply K-Means, Spectral, DBSCAN, GMM
   - Evaluate with 6+ metrics
   ↓
5. Cross-Validation
   - 5-fold stratified CV
   - Statistical validation
   ↓
6. Results & Visualization
   - Generate comparison charts
   - Create summary tables
   - Export reports
```

## Example Results

### Sample Output Table

| Algorithm | #Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz | NMI | ARI | V-Measure | Cluster Accuracy |
|-----------|-----------|------------|----------------|-------------------|-----|-----|-----------|------------------|
| K-Means | 10 | 0.41 | 0.86 | 240 | 0.52 | 0.47 | 0.55 | 0.60 |
| Spectral Clustering | 10 | 0.57 | 0.52 | 310 | 0.68 | 0.63 | 0.70 | 0.73 |
| DBSCAN | auto | 0.45 | 0.70 | 280 | 0.60 | 0.55 | 0.61 | 0.66 |
| GMM | 10 | 0.50 | 0.65 | 295 | 0.63 | 0.58 | 0.66 | 0.71 |

*Note: Actual results will vary based on the dataset and random initialization*

## Key Findings (Template)

After running the experiments, document:

1. **Best Performing Algorithm**: [Based on metrics]
2. **Optimal Train-Test Split**: [Based on stability and performance]
3. **Feature Importance**: [From PCA analysis]
4. **Cluster Characteristics**: [Genre groupings discovered]
5. **Recommendations**: [For production deployment]

## Customization

### Adjust Number of Clusters
Edit in `clustering_implementation.py`:
```python
n_clusters = 10  # Change this value
```

### Modify PCA Components
Edit in `clustering_implementation.py`:
```python
pca = PCA(n_components=20)  # Adjust components
```

### Change Cross-Validation Folds
Edit in `cross_validation.py`:
```python
n_folds = 5  # Adjust number of folds
```

### Adjust DBSCAN Parameters
Edit in `clustering_implementation.py`:
```python
eps = 2.5          # Neighborhood radius
min_samples = 5    # Minimum points per cluster
```

## Troubleshooting

### Import Errors
```bash
# Install missing packages
pip install numpy pandas scikit-learn matplotlib seaborn scipy
```

### Memory Issues
- Reduce PCA components
- Use MiniBatch K-Means instead of K-Means
- Process smaller subsets of data

### DBSCAN Finding Too Many Noise Points
- Decrease `eps` parameter
- Decrease `min_samples` parameter
- Check data scaling

## References

1. **GTZAN Dataset**: Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals.
2. **Scikit-learn Documentation**: https://scikit-learn.org/stable/
3. **Clustering Evaluation**: https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

## Project Requirements Checklist

- [x] Data adequacy check
- [x] Class balance analysis
- [x] Descriptive statistical analysis
- [x] Outlier detection with boxplots
- [x] Missing value handling
- [x] Distribution pattern identification
- [x] Percentile and quartile analysis
- [x] Trimmed statistics
- [x] Correlation analysis
- [x] Multiple clustering algorithms (K-Means, Spectral, DBSCAN, GMM)
- [x] Multiple train-test splits (50-50, 60-40, 70-30, 80-20)
- [x] Cross-validation
- [x] 6+ evaluation metrics (Silhouette, DBI, CHI, NMI, ARI, V-Measure)
- [x] PCA dimensionality reduction
- [x] Comprehensive documentation
- [x] Visualization and reporting

## Contact & Support

For questions or issues:
- Check the code documentation
- Review the TO_DO.md file for requirements
- Examine generated visualizations in results/

## License

This project is for academic purposes as part of a machine learning course.

---

**Note**: Make sure to have the GTZAN dataset properly placed in the `gtzan/` directory before running the code.
