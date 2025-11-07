"""
Cross-Validation and Advanced Evaluation Module
==============================================

This module implements cross-validation strategies and advanced
evaluation techniques for clustering algorithms.

Author: Anirudh Sharma
Topic: Unsupervised Music Genre Discovery Using Audio Feature Learning
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score, v_measure_score
)
import warnings
warnings.filterwarnings('ignore')


class CrossValidatedClusterer:
    """
    Implements cross-validation for clustering evaluation.

    This class provides k-fold cross-validation for clustering algorithms
    to ensure robust performance evaluation.

    Attributes:
        X (np.ndarray): Feature matrix
        y (np.ndarray): True labels
        n_folds (int): Number of folds for cross-validation
        scaler (StandardScaler): Feature scaler
        pca (PCA): PCA transformer
    """

    def __init__(self, X, y, n_folds=5, n_components=20):
        """
        Initialize cross-validated clusterer.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): True labels
            n_folds (int): Number of folds
            n_components (int): Number of PCA components
        """
        self.X = X
        self.y = y
        self.n_folds = n_folds
        self.n_components = n_components

        # Preprocessing
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)

        self.pca = PCA(n_components=n_components, random_state=42)
        self.X_pca = self.pca.fit_transform(self.X_scaled)

        self.n_clusters = len(np.unique(y))

        print(f"✓ CrossValidatedClusterer initialized")
        print(f"  - Samples: {len(X)}")
        print(f"  - Features: {X.shape[1]} → {n_components} (PCA)")
        print(f"  - Folds: {n_folds}")
        print(f"  - Clusters: {self.n_clusters}")

    def evaluate_algorithm(self, X_test, y_test, y_pred, algorithm_name):
        """
        Evaluate clustering algorithm.

        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): True test labels
            y_pred (np.ndarray): Predicted labels
            algorithm_name (str): Algorithm name

        Returns:
            dict: Evaluation metrics
        """
        # Handle noise points (for DBSCAN)
        valid_mask = y_pred != -1
        X_valid = X_test[valid_mask]
        y_test_valid = y_test[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        n_clusters = len(np.unique(y_pred_valid))

        if n_clusters < 2:
            return None

        try:
            metrics = {
                'Algorithm': algorithm_name,
                'Silhouette': silhouette_score(X_valid, y_pred_valid),
                'Davies_Bouldin': davies_bouldin_score(X_valid, y_pred_valid),
                'Calinski_Harabasz': calinski_harabasz_score(X_valid, y_pred_valid),
                'NMI': normalized_mutual_info_score(y_test_valid, y_pred_valid),
                'ARI': adjusted_rand_score(y_test_valid, y_pred_valid),
                'V_Measure': v_measure_score(y_test_valid, y_pred_valid)
            }
            return metrics
        except:
            return None

    def cross_validate_kmeans(self):
        """
        Cross-validate K-Means clustering.

        Returns:
            list: List of metrics for each fold
        """
        print(f"\n  Cross-validating K-Means...")

        kfold = StratifiedKFold(n_splits=self.n_folds,
                                shuffle=True, random_state=42)
        results = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(self.X_pca, self.y)):
            X_train, X_test = self.X_pca[train_idx], self.X_pca[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            model = KMeans(n_clusters=self.n_clusters,
                           random_state=42, n_init=10)
            model.fit(X_train)
            y_pred = model.predict(X_test)

            metrics = self.evaluate_algorithm(
                X_test, y_test, y_pred, 'K-Means')
            if metrics:
                metrics['Fold'] = fold + 1
                results.append(metrics)

        return results

    def cross_validate_minibatch_kmeans(self):
        """
        Cross-validate MiniBatch K-Means clustering.

        Returns:
            list: List of metrics for each fold
        """
        print(f"  Cross-validating MiniBatch K-Means...")

        kfold = StratifiedKFold(n_splits=self.n_folds,
                                shuffle=True, random_state=42)
        results = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(self.X_pca, self.y)):
            X_train, X_test = self.X_pca[train_idx], self.X_pca[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            model = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42,
                                    batch_size=100, n_init=10)
            model.fit(X_train)
            y_pred = model.predict(X_test)

            metrics = self.evaluate_algorithm(
                X_test, y_test, y_pred, 'MiniBatch K-Means')
            if metrics:
                metrics['Fold'] = fold + 1
                results.append(metrics)

        return results

    def cross_validate_spectral(self):
        """
        Cross-validate Spectral Clustering.

        Returns:
            list: List of metrics for each fold
        """
        print(f"  Cross-validating Spectral Clustering...")

        kfold = StratifiedKFold(n_splits=self.n_folds,
                                shuffle=True, random_state=42)
        results = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(self.X_pca, self.y)):
            X_test = self.X_pca[test_idx]
            y_test = self.y[test_idx]

            # Spectral clustering doesn't have separate fit/predict
            model = SpectralClustering(n_clusters=self.n_clusters, random_state=42,
                                       affinity='nearest_neighbors', n_neighbors=10)
            y_pred = model.fit_predict(X_test)

            metrics = self.evaluate_algorithm(
                X_test, y_test, y_pred, 'Spectral Clustering')
            if metrics:
                metrics['Fold'] = fold + 1
                results.append(metrics)

        return results

    def cross_validate_dbscan(self):
        """
        Cross-validate DBSCAN clustering.

        Returns:
            list: List of metrics for each fold
        """
        print(f"  Cross-validating DBSCAN...")

        kfold = StratifiedKFold(n_splits=self.n_folds,
                                shuffle=True, random_state=42)
        results = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(self.X_pca, self.y)):
            X_test = self.X_pca[test_idx]
            y_test = self.y[test_idx]

            model = DBSCAN(eps=2.5, min_samples=5)
            y_pred = model.fit_predict(X_test)

            metrics = self.evaluate_algorithm(X_test, y_test, y_pred, 'DBSCAN')
            if metrics:
                metrics['Fold'] = fold + 1
                results.append(metrics)

        return results

    def cross_validate_gmm(self):
        """
        Cross-validate Gaussian Mixture Model clustering.

        Returns:
            list: List of metrics for each fold
        """
        print(f"  Cross-validating GMM...")

        kfold = StratifiedKFold(n_splits=self.n_folds,
                                shuffle=True, random_state=42)
        results = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(self.X_pca, self.y)):
            X_train, X_test = self.X_pca[train_idx], self.X_pca[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            model = GaussianMixture(n_components=self.n_clusters, random_state=42,
                                    covariance_type='full', n_init=10)
            model.fit(X_train)
            y_pred = model.predict(X_test)

            metrics = self.evaluate_algorithm(X_test, y_test, y_pred, 'GMM')
            if metrics:
                metrics['Fold'] = fold + 1
                results.append(metrics)

        return results

    def run_all_cross_validations(self):
        """
        Run cross-validation for all algorithms.

        Returns:
            pd.DataFrame: Combined results from all algorithms
        """
        print("\n" + "=" * 80)
        print(f"RUNNING {self.n_folds}-FOLD CROSS-VALIDATION")
        print("=" * 80)

        all_results = []

        # K-Means
        results = self.cross_validate_kmeans()
        all_results.extend(results)

        # MiniBatch K-Means
        results = self.cross_validate_minibatch_kmeans()
        all_results.extend(results)

        # Spectral Clustering
        results = self.cross_validate_spectral()
        all_results.extend(results)

        # DBSCAN
        results = self.cross_validate_dbscan()
        all_results.extend(results)

        # GMM
        results = self.cross_validate_gmm()
        all_results.extend(results)

        # Create dataframe
        results_df = pd.DataFrame(all_results)

        # Save results
        results_df.to_csv('results/cross_validation_results.csv', index=False)
        print(f"\n✓ Cross-validation results saved: results/cross_validation_results.csv")

        return results_df

    def visualize_cv_results(self, results_df):
        """
        Visualize cross-validation results.

        Args:
            results_df (pd.DataFrame): Cross-validation results
        """
        print("\n" + "=" * 80)
        print("VISUALIZING CROSS-VALIDATION RESULTS")
        print("=" * 80)

        metrics = ['Silhouette', 'Davies_Bouldin', 'Calinski_Harabasz',
                   'NMI', 'ARI', 'V_Measure']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            # Box plot for each algorithm
            data_to_plot = []
            labels = []

            for algo in results_df['Algorithm'].unique():
                algo_data = results_df[results_df['Algorithm']
                                       == algo][metric].values
                data_to_plot.append(algo_data)
                labels.append(algo)

            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

            # Color boxes
            colors = plt.cm.Set3(range(len(labels)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            ax.set_title(f'{metric} (Cross-Validation)',
                         fontsize=12, fontweight='bold')
            ax.set_xlabel('Algorithm', fontweight='bold')
            ax.set_ylabel(metric, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/cross_validation_boxplots.png',
                    dpi=300, bbox_inches='tight')
        print("  ✓ Box plots saved: results/cross_validation_boxplots.png")
        plt.close()

    def generate_cv_summary(self, results_df):
        """
        Generate cross-validation summary statistics.

        Args:
            results_df (pd.DataFrame): Cross-validation results

        Returns:
            pd.DataFrame: Summary statistics
        """
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 80)

        # Calculate mean and std for each metric
        summary = results_df.groupby('Algorithm').agg({
            'Silhouette': ['mean', 'std'],
            'Davies_Bouldin': ['mean', 'std'],
            'Calinski_Harabasz': ['mean', 'std'],
            'NMI': ['mean', 'std'],
            'ARI': ['mean', 'std'],
            'V_Measure': ['mean', 'std']
        }).round(4)

        print("\nCross-Validation Results (Mean ± Std):")
        print(summary.to_string())

        # Save summary
        summary.to_csv('results/cross_validation_summary.csv')
        print(f"\n✓ Summary saved: results/cross_validation_summary.csv")

        return summary


if __name__ == "__main__":
    """
    Main execution for cross-validation.
    """
    import os
    os.makedirs('results', exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('gtzan/features_30_sec.csv')

    # Prepare data
    feature_cols = [
        col for col in df.columns if col not in ['filename', 'label']]
    X = df[feature_cols].values

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)

    # Run cross-validation
    cv_clusterer = CrossValidatedClusterer(X, y, n_folds=5, n_components=20)
    cv_results = cv_clusterer.run_all_cross_validations()
    cv_clusterer.visualize_cv_results(cv_results)
    cv_summary = cv_clusterer.generate_cv_summary(cv_results)

    print("\n✓ Cross-validation completed!")
