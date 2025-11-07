"""
Configuration file for the Music Genre Discovery Project
========================================================

This file contains all configurable parameters for the project.

Author: Anirudh Sharma
Date: November 2025
"""

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Dataset paths
DATASET_30SEC = 'gtzan/features_30_sec.csv'
DATASET_3SEC = 'gtzan/features_3_sec.csv'
DATASET_METADATA = 'gtzan/gtzan_metadata.csv'

# Output paths
OUTPUT_DIR = 'results'
CLEANED_DATASET = 'gtzan/features_30_sec_cleaned.csv'

# ============================================================================
# DATA ANALYSIS CONFIGURATION
# ============================================================================

# Outlier detection
OUTLIER_METHOD = 'iqr'  # 'iqr' or 'zscore'
IQR_THRESHOLD = 1.5     # Standard IQR multiplier
ZSCORE_THRESHOLD = 3    # Z-score threshold

# Missing value handling
FILL_METHOD = 'mean'    # 'mean', 'median', or 'drop'

# Trimmed statistics
TRIM_FRACTION = 0.1     # 10% trimming from each end

# Correlation analysis
CORRELATION_METHOD = 'pearson'  # 'pearson', 'spearman', or 'kendall'

# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================

# PCA configuration
N_PCA_COMPONENTS = 20
PCA_RANDOM_STATE = 42

# ============================================================================
# CLUSTERING CONFIGURATION
# ============================================================================

# Number of clusters (should match number of genres)
N_CLUSTERS = 10

# Random state for reproducibility
RANDOM_STATE = 42

# K-Means parameters
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300

# MiniBatch K-Means parameters
MINIBATCH_BATCH_SIZE = 100
MINIBATCH_N_INIT = 10

# Spectral Clustering parameters
SPECTRAL_AFFINITY = 'nearest_neighbors'
SPECTRAL_N_NEIGHBORS = 10

# DBSCAN parameters
DBSCAN_EPS = 2.5
DBSCAN_MIN_SAMPLES = 5

# GMM parameters
GMM_COVARIANCE_TYPE = 'full'
GMM_N_INIT = 10

# ============================================================================
# TRAIN-TEST SPLIT CONFIGURATION
# ============================================================================

# Train-test split ratios (train_size, test_size)
SPLIT_RATIOS = [
    (50, 50),
    (60, 40),
    (70, 30),
    (80, 20)
]

# ============================================================================
# CROSS-VALIDATION CONFIGURATION
# ============================================================================

# Number of folds for cross-validation
N_FOLDS = 5

# Use stratified k-fold
STRATIFIED_CV = True

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Plot style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# Figure DPI
FIGURE_DPI = 300

# Color palette
COLOR_PALETTE = 'husl'

# Figure sizes
FIGSIZE_SMALL = (10, 6)
FIGSIZE_MEDIUM = (15, 10)
FIGSIZE_LARGE = (20, 12)

# ============================================================================
# EVALUATION METRICS
# ============================================================================

# Metrics to compute
INTERNAL_METRICS = [
    'silhouette',
    'davies_bouldin',
    'calinski_harabasz'
]

EXTERNAL_METRICS = [
    'nmi',
    'ari',
    'v_measure',
    'cluster_accuracy'
]

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Verbosity level
VERBOSE = True

# Log file
LOG_FILE = 'results/experiment.log'

# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================

# Number of parallel jobs (-1 for all cores)
N_JOBS = -1

# Memory optimization
USE_MINIBATCH = True  # Use MiniBatch K-Means for large datasets

# ============================================================================
# GENRE LABELS
# ============================================================================

GENRE_NAMES = [
    'blues',
    'classical',
    'country',
    'disco',
    'hiphop',
    'jazz',
    'metal',
    'pop',
    'reggae',
    'rock'
]

# ============================================================================
# FEATURE GROUPS
# ============================================================================

# Group features for analysis
FEATURE_GROUPS = {
    'temporal': [
        'length',
        'tempo',
        'zero_crossing_rate_mean',
        'zero_crossing_rate_var'
    ],
    'spectral': [
        'spectral_centroid_mean',
        'spectral_centroid_var',
        'spectral_bandwidth_mean',
        'spectral_bandwidth_var',
        'rolloff_mean',
        'rolloff_var'
    ],
    'chroma': [
        'chroma_stft_mean',
        'chroma_stft_var'
    ],
    'energy': [
        'rms_mean',
        'rms_var'
    ],
    'harmony': [
        'harmony_mean',
        'harmony_var',
        'perceptr_mean',
        'perceptr_var'
    ]
}

# ============================================================================
# REPORT CONFIGURATION
# ============================================================================

# Generate LaTeX tables
GENERATE_LATEX = True

# Generate executive summary
GENERATE_SUMMARY = True

# Save predictions
SAVE_PREDICTIONS = True

# ============================================================================
# EXPERIMENT SETTINGS
# ============================================================================

# Run cross-validation
RUN_CROSS_VALIDATION = True

# Generate visualizations
GENERATE_VISUALIZATIONS = True

# Save intermediate results
SAVE_INTERMEDIATE = True

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_config_summary():
    """
    Print a summary of the configuration.

    Returns:
        dict: Configuration summary
    """
    summary = {
        'Dataset': DATASET_30SEC,
        'PCA Components': N_PCA_COMPONENTS,
        'Number of Clusters': N_CLUSTERS,
        'Train-Test Splits': SPLIT_RATIOS,
        'Cross-Validation Folds': N_FOLDS,
        'DBSCAN eps': DBSCAN_EPS,
        'DBSCAN min_samples': DBSCAN_MIN_SAMPLES,
        'Random State': RANDOM_STATE
    }

    print("=" * 80)
    print("CONFIGURATION SUMMARY")
    print("=" * 80)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    """
    Display configuration summary.
    """
    print("Music Genre Discovery - Configuration File")
    print()
    get_config_summary()
    print("\n✓ Configuration loaded successfully!")
