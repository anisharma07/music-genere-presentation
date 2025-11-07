# Unsupervised Music Genre Discovery Using Audio Feature Learning

This project implements a comprehensive analysis system for discovering music genres using unsupervised machine learning techniques on Spotify audio features. The system performs data preprocessing, exploratory data analysis, dimensionality reduction, clustering, and evaluation using multiple algorithms.

## 🎵 Project Overview

**Objective**: Discover hidden patterns and genre clusters in music using audio features without labeled data.

**Key Features**:
- Comprehensive data preprocessing and cleaning
- Exploratory Data Analysis with statistical measures
- Feature engineering and dimensionality reduction (PCA)
- Multiple clustering algorithms implementation
- Comprehensive evaluation using internal and external metrics
- Interactive visualizations and detailed reporting

## 📊 Dataset

The project uses Spotify audio features dataset with the following structure:

- **Main Dataset**: `Spotify/data/data.csv` (~170K tracks)
- **Genre Data**: `Spotify/data/data_w_genres.csv` (~28K tracks)
- **Additional Files**: `data_by_year.csv`, `data_by_artist.csv`, `data_by_genres.csv`

### Audio Features Analyzed:
- `acousticness`: Acoustic quality measure
- `danceability`: Rhythmic and tempo suitability for dancing
- `energy`: Perceptual measure of intensity and power
- `instrumentalness`: Prediction of instrumental content
- `liveness`: Presence of live audience
- `loudness`: Overall loudness in decibels
- `speechiness`: Presence of spoken words
- `tempo`: Track tempo in BPM
- `valence`: Musical positivity/mood

### Additional Features:
- `duration_ms`: Track duration
- `popularity`: Track popularity score
- `key`: Musical key
- `mode`: Major/minor modality
- `year`: Release year

## 🔧 Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 4GB RAM (8GB recommended for full dataset)
- 2GB free disk space for outputs

### Step 1: Clone/Download the Project

```bash
# If using git
git clone <repository-url>
cd GENERE_SPOTIFY

# Or download and extract the project files
```

### Step 2: Install Dependencies

**Option A: Using Virtual Environment (Recommended)**
```bash
# Create virtual environment
python3 -m venv music_genre_env

# Activate virtual environment
source music_genre_env/bin/activate  # On Linux/Mac
# OR
music_genre_env\Scripts\activate     # On Windows

# Install required packages
pip install -r requirements.txt
```

**Option B: System-wide Installation**
```bash
# Install required packages (may need --user flag)
pip install --user -r requirements.txt

# Or with system packages override (not recommended)
pip install --break-system-packages -r requirements.txt
```

**Required packages** (automatically installed):
```
pandas>=2.0.0          # Data manipulation and analysis
numpy>=1.24.0          # Numerical computing
scikit-learn>=1.3.0    # Machine learning algorithms
matplotlib>=3.7.0      # Basic plotting
seaborn>=0.12.0        # Statistical visualizations
plotly>=5.15.0         # Interactive visualizations
scipy>=1.10.0          # Statistical functions
yellowbrick>=1.5.0     # ML visualization
librosa>=0.10.0        # Audio analysis (optional)
jupyter>=1.0.0         # Jupyter notebook support
ipykernel>=6.0.0       # Jupyter kernel
```

### Step 3: Verify Data Structure

Ensure your data is organized as follows:
```
GENERE_SPOTIFY/
├── Spotify/
│   └── data/
│       ├── data.csv                 # Main dataset (required)
│       ├── data_w_genres.csv        # Genre-labeled data
│       ├── data_by_year.csv         # Yearly aggregated data
│       ├── data_by_artist.csv       # Artist-based data
│       └── data_by_genres.csv       # Genre-based data
├── music_genre_analysis.py          # Main analysis module
├── run_analysis.py                  # Runner script
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## 🚀 Usage

### Quick Start (Recommended)

For a quick analysis with reduced dataset:
```bash
python run_analysis.py --quick
```

### Full Analysis

For complete analysis with all data:
```bash
python run_analysis.py --full
```

### Comprehensive Experiments

To run experiments with multiple configurations:
```bash
python run_analysis.py --experiments
```

### Custom Configuration

```bash
# Specify number of clusters
python run_analysis.py --clusters 15

# Quick analysis with custom clusters
python run_analysis.py --quick --clusters 8

# Full analysis with experiments
python run_analysis.py --full --experiments
```

### Advanced Usage

For advanced usage, import the module directly:

```python
from music_genre_analysis import MusicGenreAnalyzer

# Initialize analyzer
analyzer = MusicGenreAnalyzer()

# Load and preprocess data
data = analyzer.load_and_preprocess_data()

# Run specific components
analyzer.exploratory_data_analysis()
analyzer.create_visualizations()
analyzer.prepare_features()
analyzer.perform_clustering(n_clusters=12)
analyzer.evaluate_clustering()
```

## 📈 Algorithms Implemented

### 1. Clustering Algorithms

1. **K-Means Clustering**
   - Partitional clustering using centroids
   - Good for spherical clusters
   - Parameters: n_clusters=10 (default)

2. **MiniBatch K-Means**
   - Scalable variant of K-Means
   - Faster computation for large datasets
   - Similar performance to K-Means

3. **Spectral Clustering**
   - Graph-based clustering method
   - Good for non-spherical clusters
   - Uses eigenvalues of similarity matrix

4. **DBSCAN (Density-Based Spatial Clustering)**
   - Density-based clustering
   - Automatically determines number of clusters
   - Good for irregular cluster shapes
   - Parameters: eps=0.5, min_samples=5

5. **Gaussian Mixture Model (GMM)**
   - Probabilistic clustering approach
   - Assumes data comes from mixture of Gaussians
   - Provides soft cluster assignments

### 2. Evaluation Metrics

#### Internal Metrics (No ground truth required):
- **Silhouette Score**: Measures cluster cohesion and separation (-1 to 1, higher is better)
- **Davies-Bouldin Index**: Average similarity ratio of clusters (lower is better)
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster dispersion (higher is better)

#### External Metrics (Using decade as proxy ground truth):
- **Adjusted Rand Index (ARI)**: Similarity between predicted and true clusters
- **Normalized Mutual Information (NMI)**: Mutual information between clusterings
- **V-Measure**: Harmonic mean of homogeneity and completeness
- **Purity Index**: Fraction of correctly classified instances

## 📊 Output Files

After running the analysis, the following outputs are generated:

### 1. Visualizations (`visualizations/`)
- `feature_distributions.png`: Distribution of audio features
- `box_plots.png`: Box plots for outlier identification
- `correlation_heatmap.png`: Feature correlation matrix
- `pairplot.png`: Pairwise feature relationships
- `decade_evolution.png`: Feature evolution over time

### 2. Clustering Results (`clustering_results/`)
- `clustering_pca_visualization.png`: PCA visualization of clusters
- `silhouette_comparison.png`: Algorithm performance comparison
- `metrics_heatmap.png`: Comprehensive metrics heatmap

### 3. Numerical Results (`results/`)
- `evaluation_metrics.csv`: Complete evaluation results
- `basic_statistics.csv`: Descriptive statistics
- `clustering_results.json`: Cluster assignments
- `processed_features.npy`: Preprocessed feature matrix

### 4. Report
- `analysis_report.html`: Comprehensive HTML report with findings

## 🔬 Analysis Pipeline

### 1. Data Preprocessing
- **Outlier Removal**: IQR method with 1.5× threshold
- **Missing Value Handling**: Mean imputation
- **Duplicate Removal**: Based on track name and artist
- **Feature Engineering**: Duration conversion, decade creation

### 2. Exploratory Data Analysis
- Descriptive statistics (mean, median, quartiles)
- Distribution analysis and normality testing
- Correlation analysis
- Temporal evolution of features
- Statistical measures as per requirements:
  - Sample mean (X̄)
  - 25th and 75th percentiles
  - Median (M) and third quartile (Q3)
  - Trimmed mean (X̄T) and trimmed standard deviation (ST)

### 3. Feature Engineering
- **Standardization**: StandardScaler for zero mean, unit variance
- **Dimensionality Reduction**: PCA to 20 components
- **Feature Selection**: Focus on audio characteristics

### 4. Clustering Analysis
- Multiple algorithm implementation
- Hyperparameter optimization
- Cross-validation with different train/test splits (50-50, 60-40, 70-30, 80-20)

### 5. Evaluation and Comparison
- Comprehensive metric calculation
- Statistical significance testing
- Visualization and interpretation

## 📊 Expected Results

### Sample Output Table:
| Algorithm | #Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz | NMI | ARI | V-Measure |
|-----------|-----------|------------|----------------|-------------------|-----|-----|-----------|
| K-Means | 10 | 0.412 | 0.863 | 240.5 | 0.523 | 0.471 | 0.552 |
| Spectral | 10 | 0.574 | 0.521 | 310.2 | 0.681 | 0.634 | 0.701 |
| DBSCAN | auto | 0.451 | 0.702 | 280.1 | 0.603 | 0.554 | 0.614 |
| GMM | 10 | 0.501 | 0.654 | 295.3 | 0.634 | 0.584 | 0.661 |

### Performance Insights:
- **Spectral Clustering** typically performs best for music features
- **DBSCAN** good for identifying core genre clusters with noise
- **K-Means** provides baseline performance
- **GMM** offers probabilistic cluster membership

## 🛠 Troubleshooting

### Common Issues:

1. **Memory Errors**
   ```bash
   # Use quick analysis for large datasets
   python run_analysis.py --quick
   ```

2. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install --upgrade -r requirements.txt
   ```

3. **Data File Not Found**
   - Verify data is in `Spotify/data/` directory
   - Check file names match expected format
   - Ensure CSV files are not corrupted

4. **Visualization Issues**
   ```bash
   # For systems without GUI (servers)
   export MPLBACKEND=Agg
   python run_analysis.py
   ```

5. **Slow Performance**
   - Use `--quick` flag for initial testing
   - Consider sampling large datasets
   - Increase system memory if possible

### Performance Optimization:
- **Quick Analysis**: ~2-5 minutes (5K samples)
- **Full Analysis**: ~15-30 minutes (170K samples)
- **With Experiments**: ~45-90 minutes (multiple configurations)

## 📝 Customization

### Modify Clustering Parameters:

```python
# In music_genre_analysis.py
analyzer = MusicGenreAnalyzer()

# Custom DBSCAN parameters
analyzer.perform_clustering_custom({
    'DBSCAN': DBSCAN(eps=0.3, min_samples=10)
})

# Different PCA components
analyzer.pca = PCA(n_components=30)
```

### Add New Features:

```python
# Extend feature list
analyzer.audio_features.extend(['new_feature1', 'new_feature2'])

# Custom feature engineering
def custom_feature_engineering(self, data):
    data['energy_valence_ratio'] = data['energy'] / (data['valence'] + 0.001)
    return data
```

### Custom Evaluation Metrics:

```python
# Add custom metrics to evaluation
def custom_metric(true_labels, pred_labels):
    # Implementation
    return score

# Include in evaluation pipeline
```

## 🎯 Research Applications

This codebase supports various research directions:

1. **Genre Evolution Studies**: Analyze how musical characteristics change over decades
2. **Cross-Cultural Music Analysis**: Compare features across different regions
3. **Recommendation Systems**: Use clusters for music recommendation
4. **Feature Importance**: Identify key features for genre classification
5. **Ensemble Methods**: Combine multiple clustering results

## 📚 References

- **Spotify Web API**: Audio features documentation
- **Scikit-learn**: Machine learning algorithms
- **Music Information Retrieval**: Academic research in MIR
- **Unsupervised Learning**: Clustering techniques and evaluation

## 👥 Contributing

To extend this project:

1. Fork the repository
2. Add new features or algorithms
3. Update documentation
4. Submit pull request with tests

## 📄 License

This project is for educational and research purposes. Please cite appropriately if used in academic work.

## 🔗 Additional Resources

- **Jupyter Notebooks**: Create interactive analysis notebooks
- **Web Interface**: Build Flask/Django web interface
- **API Integration**: Direct Spotify API integration
- **Advanced Visualization**: Interactive Plotly dashboards
- **Deep Learning**: Neural network-based clustering approaches

---

**For support or questions**, please refer to the code comments or create an issue in the project repository.

**Happy Music Analysis! 🎵📊**