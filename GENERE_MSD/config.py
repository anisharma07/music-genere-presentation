"""
Configuration file for Music Genre Discovery Project
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "million song",
                        "millionsongsubset", "MillionSongSubset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# Create output directories
for directory in [OUTPUT_DIR, RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Feature extraction settings
FEATURE_CONFIG = {
    'use_existing_features': True,  # Use features from HDF5 files
    'extract_additional_features': False,  # Set to True if you have audio files
}

# PCA settings
PCA_COMPONENTS = 20

# Clustering settings
CLUSTERING_CONFIG = {
    'kmeans': {
        'n_clusters': [5, 10, 15, 20],
        'random_state': 42,
        'max_iter': 300,
        'n_init': 10
    },
    'minibatch_kmeans': {
        'n_clusters': [5, 10, 15, 20],
        'random_state': 42,
        'max_iter': 300,
        'batch_size': 100
    },
    'spectral': {
        'n_clusters': [5, 10, 15, 20],
        'random_state': 42,
        'n_init': 10,
        'affinity': 'nearest_neighbors'
    },
    'dbscan': {
        'eps': [0.5, 1.0, 1.5, 2.0],
        'min_samples': [5, 10, 15],
        'metric': 'euclidean'
    },
    'gmm': {
        'n_components': [5, 10, 15, 20],
        'random_state': 42,
        'covariance_type': 'full',
        'max_iter': 100
    }
}

# Train-Test split ratios
SPLIT_RATIOS = [
    (0.5, 0.5),  # 50-50
    (0.6, 0.4),  # 60-40
    (0.7, 0.3),  # 70-30
    (0.8, 0.2),  # 80-20
]

# Cross-validation folds
CV_FOLDS = 5

# Random seed for reproducibility
RANDOM_SEED = 42

# Evaluation metrics
INTERNAL_METRICS = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
EXTERNAL_METRICS = ['adjusted_rand', 'normalized_mutual_info', 'v_measure']

# Visualization settings
PLOT_SETTINGS = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8-darkgrid'
}

# Data cleaning settings
DATA_CLEANING = {
    'remove_outliers': True,
    'outlier_method': 'iqr',  # 'iqr' or 'zscore'
    'iqr_multiplier': 1.5,
    'zscore_threshold': 3,
    'handle_missing': 'mean',  # 'mean', 'median', 'drop'
    'trimming_fraction': 0.05  # For trimmed statistics
}

# Number of workers for parallel processing
N_JOBS = -1  # -1 means use all available cores

# Logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}
