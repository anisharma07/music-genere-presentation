"""
PROJECT DOCUMENTATION
====================

Unsupervised Music Genre Discovery Using Audio Feature Learning
GTZAN Dataset Analysis

Author: Anirudh Sharma
Date: November 2025

This document provides comprehensive documentation for the entire project.
"""

# ============================================================================
# TABLE OF CONTENTS
# ============================================================================

"""
1. PROJECT OVERVIEW
2. DATASET DESCRIPTION
3. METHODOLOGY
4. MODULE DESCRIPTIONS
5. ALGORITHMS IMPLEMENTED
6. EVALUATION METRICS
7. WORKFLOW
8. RESULTS INTERPRETATION
9. CODE EXAMPLES
10. TROUBLESHOOTING
11. REFERENCES
"""

# ============================================================================
# 1. PROJECT OVERVIEW
# ============================================================================

"""
PROJECT GOAL:
Discover music genres in an unsupervised manner using audio features,
without relying on labeled data during clustering.

KEY OBJECTIVES:
- Analyze and clean the GTZAN dataset
- Apply dimensionality reduction (PCA)
- Implement 4+ clustering algorithms
- Evaluate using 6+ metrics (internal and external)
- Compare performance across different train-test splits
- Validate results using cross-validation
- Generate comprehensive visualizations and reports

EXPECTED OUTCOMES:
- Identify natural groupings in music data
- Determine best clustering algorithm for music genre discovery
- Understand feature importance through PCA
- Document performance metrics and comparisons
"""

# ============================================================================
# 2. DATASET DESCRIPTION
# ============================================================================

"""
GTZAN DATASET:

Source: Music Genre Classification Dataset
Total Samples: ~1000 audio tracks
Duration: 30 seconds each
Sample Rate: 22050 Hz
Genres: 10 (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
Samples per Genre: ~100

FEATURES EXTRACTED:
1. Temporal Features:
   - Length
   - Tempo (BPM)
   - Zero Crossing Rate (mean, variance)

2. Spectral Features:
   - Spectral Centroid (mean, variance)
   - Spectral Bandwidth (mean, variance)
   - Spectral Rolloff (mean, variance)

3. Chroma Features:
   - Chroma STFT (mean, variance)

4. Energy Features:
   - RMS Energy (mean, variance)

5. Harmonic Features:
   - Harmony (mean, variance)
   - Perceptual (mean, variance)

6. MFCCs:
   - 20 MFCC coefficients (mean, variance)
   - Total: 40 MFCC features

TOTAL FEATURES: ~57 features per sample

DATA FILES:
- features_30_sec.csv: Features from 30-second clips
- features_3_sec.csv: Features from 3-second clips
- gtzan_metadata.csv: Metadata (filename, genre, duration)
"""

# ============================================================================
# 3. METHODOLOGY
# ============================================================================

"""
COMPLETE WORKFLOW:

PHASE 1: DATA ANALYSIS AND CLEANING
├── 1.1 Data Adequacy Check
│   └── Verify sufficient samples and features
├── 1.2 Class Balance Analysis
│   └── Check genre distribution
├── 1.3 Descriptive Statistics
│   └── Mean, std, skewness, kurtosis
├── 1.4 Missing Value Analysis
│   └── Detect and handle null values
├── 1.5 Outlier Detection
│   └── IQR method with boxplots
├── 1.6 Outlier Removal
│   └── Clean dataset creation
├── 1.7 Distribution Analysis
│   └── Normality tests, histograms
├── 1.8 Percentile Analysis
│   └── Quartiles, IQR
├── 1.9 Trimmed Statistics
│   └── Robust statistics
└── 1.10 Correlation Analysis
    └── Feature correlations

PHASE 2: FEATURE PREPROCESSING
├── 2.1 Feature Scaling
│   └── StandardScaler (Z-score normalization)
└── 2.2 Dimensionality Reduction
    └── PCA to 20 components (~95% variance)

PHASE 3: CLUSTERING EXPERIMENTS
├── 3.1 K-Means Clustering
├── 3.2 MiniBatch K-Means
├── 3.3 Spectral Clustering
├── 3.4 DBSCAN
└── 3.5 Gaussian Mixture Model

PHASE 4: EVALUATION
├── 4.1 Internal Metrics
│   ├── Silhouette Score
│   ├── Davies-Bouldin Index
│   └── Calinski-Harabasz Index
├── 4.2 External Metrics
│   ├── Normalized Mutual Information
│   ├── Adjusted Rand Index
│   ├── V-Measure
│   └── Cluster Accuracy
└── 4.3 Cross-Validation
    └── 5-fold stratified CV

PHASE 5: VISUALIZATION AND REPORTING
├── 5.1 Performance Comparisons
├── 5.2 Radar Charts
├── 5.3 2D Cluster Visualizations
├── 5.4 Statistical Tables
└── 5.5 Executive Summary
"""

# ============================================================================
# 4. MODULE DESCRIPTIONS
# ============================================================================

"""
MODULE: data_analysis.py
========================
Purpose: Comprehensive data analysis and cleaning

Key Classes:
- MusicDataAnalyzer: Main analysis class

Key Methods:
- check_data_adequacy(): Validate dataset
- check_class_balance(): Analyze genre distribution
- descriptive_statistics(): Statistical summary
- detect_outliers_iqr(): Outlier detection
- handle_missing_values(): Missing data handling
- remove_outliers(): Data cleaning
- analyze_distribution(): Distribution patterns
- correlation_analysis(): Feature correlations
- generate_full_report(): Complete analysis

Usage:
    analyzer = MusicDataAnalyzer('gtzan/features_30_sec.csv')
    report = analyzer.generate_full_report()

---

MODULE: clustering_implementation.py
====================================
Purpose: Clustering algorithms and evaluation

Key Classes:
- MusicGenreClusterer: Main clustering class

Key Methods:
- kmeans_clustering(): K-Means implementation
- spectral_clustering(): Spectral clustering
- dbscan_clustering(): DBSCAN implementation
- gmm_clustering(): GMM implementation
- evaluate_clustering(): Metric calculation
- run_experiment(): Single experiment
- run_all_experiments(): All train-test splits
- visualize_results(): Generate plots
- visualize_clusters_2d(): 2D PCA visualization

Usage:
    clusterer = MusicGenreClusterer('gtzan/features_30_sec_cleaned.csv')
    results = clusterer.run_all_experiments()

---

MODULE: cross_validation.py
===========================
Purpose: Cross-validation for robust evaluation

Key Classes:
- CrossValidatedClusterer: CV implementation

Key Methods:
- cross_validate_kmeans(): K-Means CV
- cross_validate_spectral(): Spectral CV
- cross_validate_dbscan(): DBSCAN CV
- cross_validate_gmm(): GMM CV
- run_all_cross_validations(): All algorithms
- visualize_cv_results(): CV visualization
- generate_cv_summary(): Summary statistics

Usage:
    cv_clusterer = CrossValidatedClusterer(X, y, n_folds=5)
    cv_results = cv_clusterer.run_all_cross_validations()

---

MODULE: utils.py
================
Purpose: Utility functions for analysis

Key Functions:
- create_comparison_table(): Format results
- plot_metric_heatmap(): Heatmap visualization
- generate_latex_table(): LaTeX export
- plot_pca_variance(): PCA analysis
- create_executive_summary(): Report generation
- export_best_model_predictions(): Save predictions
- plot_confusion_matrix_style(): Cluster mapping

---

MODULE: config.py
=================
Purpose: Centralized configuration

Key Parameters:
- N_CLUSTERS: Number of clusters (10)
- N_PCA_COMPONENTS: PCA dimensions (20)
- SPLIT_RATIOS: Train-test splits
- DBSCAN_EPS: DBSCAN epsilon
- DBSCAN_MIN_SAMPLES: DBSCAN min samples
- N_FOLDS: CV folds (5)

---

MODULE: main.py
===============
Purpose: Orchestrate complete pipeline

Execution Flow:
1. Create directories
2. Run data analysis
3. Run clustering experiments
4. Generate visualizations
5. Create reports
"""

# ============================================================================
# 5. ALGORITHMS IMPLEMENTED
# ============================================================================

"""
1. K-MEANS CLUSTERING
   Type: Partitioning
   Approach: Minimize within-cluster variance
   Pros:
     + Simple and fast
     + Scalable to large datasets
     + Works well with spherical clusters
   Cons:
     - Requires predefined K
     - Sensitive to initialization
     - Assumes equal cluster sizes
   Parameters:
     - n_clusters: 10
     - n_init: 10
     - random_state: 42

2. MINIBATCH K-MEANS
   Type: Partitioning (optimized)
   Approach: K-Means with mini-batches
   Pros:
     + Faster than K-Means
     + Better for large datasets
     + Lower memory usage
   Cons:
     - Slightly less accurate than K-Means
   Parameters:
     - n_clusters: 10
     - batch_size: 100
     - n_init: 10

3. SPECTRAL CLUSTERING
   Type: Graph-based
   Approach: Graph cut minimization
   Pros:
     + Handles non-convex clusters
     + Works with complex shapes
     + No centroid assumption
   Cons:
     - Computationally expensive
     - Memory intensive
     - Requires K
   Parameters:
     - n_clusters: 10
     - affinity: 'nearest_neighbors'
     - n_neighbors: 10

4. DBSCAN
   Type: Density-based
   Approach: Density-connected components
   Pros:
     + Discovers arbitrary shapes
     + Automatic K detection
     + Handles noise (outliers)
   Cons:
     - Sensitive to parameters
     - Struggles with varying densities
     - Poor with high dimensions
   Parameters:
     - eps: 2.5
     - min_samples: 5

5. GAUSSIAN MIXTURE MODEL (GMM)
   Type: Probabilistic
   Approach: Mixture of Gaussians
   Pros:
     + Soft clustering (probabilities)
     + Flexible cluster shapes
     + Statistical foundation
   Cons:
     - Requires K
     - Computationally expensive
     - May converge to local optima
   Parameters:
     - n_components: 10
     - covariance_type: 'full'
     - n_init: 10
"""

# ============================================================================
# 6. EVALUATION METRICS
# ============================================================================

"""
INTERNAL METRICS (Unsupervised - No labels needed)
===================================================

1. SILHOUETTE SCORE
   Formula: s(i) = (b(i) - a(i)) / max(a(i), b(i))
   Range: [-1, 1]
   Interpretation:
     > 0.7: Strong structure
     0.5-0.7: Reasonable structure
     0.25-0.5: Weak structure
     < 0.25: No substantial structure
   Best: HIGHER
   
2. DAVIES-BOULDIN INDEX
   Formula: DB = (1/k) Σ max_j≠i (σ_i + σ_j) / d(c_i, c_j)
   Range: [0, ∞]
   Interpretation:
     < 1.0: Good separation
     1.0-2.0: Moderate separation
     > 2.0: Poor separation
   Best: LOWER
   
3. CALINSKI-HARABASZ INDEX
   Formula: CH = (B_SS / (k-1)) / (W_SS / (n-k))
   Range: [0, ∞]
   Interpretation:
     Higher values = better-defined clusters
     Ratio of between to within variance
   Best: HIGHER

EXTERNAL METRICS (Supervised - Uses ground truth)
==================================================

4. NORMALIZED MUTUAL INFORMATION (NMI)
   Formula: NMI = 2 * I(U;V) / (H(U) + H(V))
   Range: [0, 1]
   Interpretation:
     1.0: Perfect agreement
     0.0: No agreement
   Best: HIGHER
   
5. ADJUSTED RAND INDEX (ARI)
   Formula: ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
   Range: [-1, 1]
   Interpretation:
     1.0: Perfect agreement
     0.0: Random labeling
     < 0: Worse than random
   Best: HIGHER
   
6. V-MEASURE
   Formula: V = 2 * (h * c) / (h + c)
   Range: [0, 1]
   Components:
     h: Homogeneity (all clusters contain single class)
     c: Completeness (all class members in same cluster)
   Best: HIGHER
   
7. CLUSTER ACCURACY
   Method: Hungarian algorithm for optimal label assignment
   Range: [0, 1]
   Interpretation:
     Percentage of correctly assigned samples
   Best: HIGHER
"""

# ============================================================================
# 7. WORKFLOW EXAMPLE
# ============================================================================

"""
STEP-BY-STEP EXECUTION:

# Step 1: Data Analysis
from data_analysis import MusicDataAnalyzer

analyzer = MusicDataAnalyzer('gtzan/features_30_sec.csv')
report = analyzer.generate_full_report()

# Outputs:
# - results/class_balance.png
# - results/descriptive_statistics.csv
# - results/outlier_boxplots.png
# - results/correlation_heatmap.png
# - gtzan/features_30_sec_cleaned.csv

# Step 2: Clustering
from clustering_implementation import MusicGenreClusterer

clusterer = MusicGenreClusterer('gtzan/features_30_sec_cleaned.csv')
results = clusterer.run_all_experiments()

# Outputs:
# - results/clustering_results.csv
# - results/summary_table.csv
# - results/metrics_comparison.png
# - results/radar_chart.png
# - results/cluster_viz_*.png

# Step 3: Cross-Validation
from cross_validation import CrossValidatedClusterer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv('gtzan/features_30_sec_cleaned.csv')
X = df[[col for col in df.columns if col not in ['filename', 'label']]].values
le = LabelEncoder()
y = le.fit_transform(df['label'].values)

cv_clusterer = CrossValidatedClusterer(X, y, n_folds=5)
cv_results = cv_clusterer.run_all_cross_validations()

# Outputs:
# - results/cross_validation_results.csv
# - results/cross_validation_summary.csv
# - results/cross_validation_boxplots.png

# Step 4: Generate Additional Reports
from utils import create_executive_summary, plot_metric_heatmap

create_executive_summary(results)
plot_metric_heatmap(results)

# Outputs:
# - results/executive_summary.txt
# - results/metric_heatmap.png
"""

# ============================================================================
# 8. RESULTS INTERPRETATION
# ============================================================================

"""
HOW TO INTERPRET RESULTS:

1. COMPARING ALGORITHMS:
   - Look at multiple metrics, not just one
   - Higher Silhouette + Lower Davies-Bouldin = Good internal structure
   - Higher NMI/ARI = Good agreement with ground truth
   - Check consistency across different splits

2. IDENTIFYING BEST ALGORITHM:
   - Create normalized scores for all metrics
   - Average across metrics
   - Consider computational cost
   - Evaluate stability (cross-validation)

3. UNDERSTANDING CLUSTERS:
   - Examine 2D visualizations
   - Check cluster-genre mapping confusion matrix
   - Analyze feature importance (PCA loadings)
   - Identify which genres cluster together

4. VALIDATION:
   - Cross-validation reduces overfitting
   - Low std deviation = stable performance
   - Compare train vs. test performance
   - Check for reasonable cluster sizes

5. PRACTICAL CONSIDERATIONS:
   - Spectral Clustering: Best quality, slowest
   - K-Means: Fast, good baseline
   - DBSCAN: Finds outliers, variable K
   - GMM: Probabilistic, flexible shapes
"""

# ============================================================================
# 9. CODE EXAMPLES
# ============================================================================

"""
EXAMPLE 1: Custom Analysis
---------------------------
from data_analysis import MusicDataAnalyzer

# Load and analyze
analyzer = MusicDataAnalyzer('gtzan/features_30_sec.csv')

# Check specific aspects
balance = analyzer.check_class_balance()
outliers = analyzer.check_outliers()
correlation = analyzer.correlation_analysis(method='spearman')

# Custom cleaning
cleaned_df = analyzer.remove_outliers(method='zscore', threshold=3)


EXAMPLE 2: Single Clustering Run
---------------------------------
from clustering_implementation import MusicGenreClusterer

clusterer = MusicGenreClusterer('gtzan/features_30_sec.csv')

# Run single experiment
results, models, X_test, y_test = clusterer.run_experiment((80, 20))

# Access specific model
kmeans_model, kmeans_labels = models['K-Means']

# Visualize
clusterer.visualize_clusters_2d(X_test, y_test, kmeans_labels, 'K-Means')


EXAMPLE 3: Custom Metric Calculation
------------------------------------
from sklearn.metrics import silhouette_score
import numpy as np

# Manually calculate metric
y_pred = kmeans_model.predict(X_test)
score = silhouette_score(X_test, y_pred)
print(f"Silhouette Score: {score:.4f}")


EXAMPLE 4: Export Results
--------------------------
from utils import export_best_model_predictions

# Save predictions
export_best_model_predictions(
    clusterer, 
    kmeans_model, 
    kmeans_labels, 
    'K-Means',
    'results/kmeans_predictions.csv'
)
"""

# ============================================================================
# 10. TROUBLESHOOTING
# ============================================================================

"""
COMMON ISSUES AND SOLUTIONS:

1. IMPORT ERRORS
   Problem: ModuleNotFoundError: No module named 'sklearn'
   Solution: pip install scikit-learn

2. MEMORY ERRORS
   Problem: MemoryError or system slowdown
   Solutions:
   - Reduce PCA components: N_PCA_COMPONENTS = 10
   - Use MiniBatch K-Means
   - Process fewer splits at once
   - Reduce dataset size

3. DBSCAN ISSUES
   Problem: DBSCAN finds only 1 cluster or all noise
   Solutions:
   - Adjust eps: Try 1.5, 2.0, 2.5, 3.0
   - Adjust min_samples: Try 3, 5, 10
   - Check data scaling
   - Visualize nearest neighbor distances

4. POOR CLUSTERING RESULTS
   Problem: Low silhouette scores, poor NMI
   Solutions:
   - Check PCA variance retained
   - Try different number of components
   - Verify data cleaning
   - Check for data leakage
   - Normalize features properly

5. VISUALIZATION ISSUES
   Problem: Plots not showing or empty
   Solutions:
   - Check results/ directory exists
   - Verify matplotlib backend
   - Check file permissions
   - Use plt.show() for interactive display

6. SLOW EXECUTION
   Problem: Code takes too long
   Solutions:
   - Use MiniBatch K-Means
   - Reduce cross-validation folds
   - Skip Spectral Clustering (slowest)
   - Use parallel processing (N_JOBS=-1)
"""

# ============================================================================
# 11. REFERENCES
# ============================================================================

"""
ACADEMIC REFERENCES:

1. GTZAN Dataset:
   Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals.
   IEEE Transactions on speech and audio processing, 10(5), 293-302.

2. Clustering Algorithms:
   - K-Means: MacQueen, J. (1967). Some methods for classification and analysis of 
     multivariate observations.
   - DBSCAN: Ester, M., et al. (1996). A density-based algorithm for discovering 
     clusters in large spatial databases with noise.
   - Spectral: Ng, A., et al. (2001). On spectral clustering: Analysis and an algorithm.
   - GMM: Reynolds, D. A. (2009). Gaussian mixture models.

3. Evaluation Metrics:
   - Silhouette: Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the 
     interpretation and validation of cluster analysis.
   - NMI: Strehl, A., & Ghosh, J. (2002). Cluster ensembles.
   - ARI: Hubert, L., & Arabie, P. (1985). Comparing partitions.

4. PCA:
   Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review 
   and recent developments.

DOCUMENTATION:

- Scikit-learn: https://scikit-learn.org/stable/
- NumPy: https://numpy.org/doc/
- Pandas: https://pandas.pydata.org/docs/
- Matplotlib: https://matplotlib.org/stable/contents.html
- Seaborn: https://seaborn.pydata.org/

TUTORIALS:

- Clustering: https://scikit-learn.org/stable/modules/clustering.html
- Evaluation: https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
- PCA: https://scikit-learn.org/stable/modules/decomposition.html#pca
"""

# ============================================================================
# END OF DOCUMENTATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PROJECT DOCUMENTATION LOADED")
    print("=" * 80)
    print("\nThis file contains comprehensive documentation for the project.")
    print("Review the docstrings for detailed information about:")
    print("  - Project overview and objectives")
    print("  - Dataset description")
    print("  - Methodology and workflow")
    print("  - Module descriptions")
    print("  - Algorithm details")
    print("  - Evaluation metrics")
    print("  - Code examples")
    print("  - Troubleshooting")
    print("  - References")
    print("=" * 80)
