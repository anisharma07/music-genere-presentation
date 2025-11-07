"""
Clustering Implementation Module for Music Genre Discovery
==========================================================

This module implements multiple unsupervised clustering algorithms:
- K-Means / MiniBatch K-Means
- Spectral Clustering
- DBSCAN (Density-Based Spatial Clustering)
- Gaussian Mixture Model (GMM)

With comprehensive evaluation using multiple metrics and visualization.

Author: Anirudh Sharma
Topic: Unsupervised Music Genre Discovery Using Audio Feature Learning
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score, v_measure_score,
    confusion_matrix, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class MusicGenreClusterer:
    """
    Comprehensive clustering system for music genre discovery.

    This class implements multiple clustering algorithms and evaluates them
    using both internal and external metrics.

    Attributes:
        df (pd.DataFrame): The dataset
        features (np.ndarray): Scaled feature matrix
        labels_true (np.ndarray): Ground truth labels
        pca_features (np.ndarray): PCA-reduced features
        scaler (StandardScaler): Feature scaler
        pca (PCA): PCA transformer
        n_genres (int): Number of unique genres
    """

    def __init__(self, filepath, use_cleaned=True):
        """
        Initialize the clusterer with a dataset.

        Args:
            filepath (str): Path to the CSV file
            use_cleaned (bool): Whether to use cleaned dataset
        """
        print("=" * 80)
        print("Initializing Music Genre Clusterer")
        print("=" * 80)

        # Load dataset
        if use_cleaned and 'cleaned' in filepath:
            self.df = pd.read_csv(filepath)
            print("✓ Using cleaned dataset")
        else:
            self.df = pd.read_csv(filepath)
            print("✓ Using original dataset")

        # Prepare features and labels
        self.label_col = 'label'
        feature_cols = [col for col in self.df.columns
                        if col not in ['filename', 'label']]

        self.features_raw = self.df[feature_cols].values
        self.labels_true = self.df[self.label_col].values
        self.n_genres = len(np.unique(self.labels_true))

        # Encode labels to integers
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(
            self.labels_true)

        # Scale features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features_raw)

        # PCA reduction
        self.pca = PCA(n_components=20, random_state=42)
        self.pca_features = self.pca.fit_transform(self.features)

        print(f"\n  - Total samples: {len(self.df)}")
        print(f"  - Total features: {self.features.shape[1]}")
        print(f"  - PCA components: {self.pca_features.shape[1]}")
        print(f"  - Number of genres: {self.n_genres}")
        print(
            f"  - PCA explained variance: {self.pca.explained_variance_ratio_.sum():.2%}")
        print()

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.

        Args:
            test_size (float): Proportion of test set
            random_state (int): Random seed

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            self.pca_features, self.labels_encoded,
            test_size=test_size, random_state=random_state,
            stratify=self.labels_encoded
        )

    def calculate_cluster_accuracy(self, labels_true, labels_pred):
        """
        Calculate cluster accuracy by finding best label mapping.

        Args:
            labels_true (np.ndarray): Ground truth labels
            labels_pred (np.ndarray): Predicted cluster labels

        Returns:
            float: Cluster accuracy (0-1)
        """
        from scipy.optimize import linear_sum_assignment

        # Create confusion matrix
        cm = confusion_matrix(labels_true, labels_pred)

        # Find optimal assignment using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-cm)

        # Calculate accuracy
        accuracy = cm[row_ind, col_ind].sum() / cm.sum()

        return accuracy

    def evaluate_clustering(self, X, labels_true, labels_pred, algorithm_name):
        """
        Evaluate clustering using multiple metrics.

        Args:
            X (np.ndarray): Feature matrix
            labels_true (np.ndarray): Ground truth labels
            labels_pred (np.ndarray): Predicted cluster labels
            algorithm_name (str): Name of the algorithm

        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Handle DBSCAN noise points (-1 labels)
        valid_mask = labels_pred != -1
        X_valid = X[valid_mask]
        labels_true_valid = labels_true[valid_mask]
        labels_pred_valid = labels_pred[valid_mask]

        # Check if we have valid clusters
        n_clusters = len(np.unique(labels_pred_valid))

        if n_clusters < 2:
            print(
                f"  ⚠ Warning: {algorithm_name} found only {n_clusters} cluster(s)")
            return None

        # Internal metrics (unsupervised)
        try:
            silhouette = silhouette_score(X_valid, labels_pred_valid)
        except:
            silhouette = -1

        try:
            davies_bouldin = davies_bouldin_score(X_valid, labels_pred_valid)
        except:
            davies_bouldin = -1

        try:
            calinski_harabasz = calinski_harabasz_score(
                X_valid, labels_pred_valid)
        except:
            calinski_harabasz = -1

        # External metrics (supervised - using ground truth)
        try:
            nmi = normalized_mutual_info_score(
                labels_true_valid, labels_pred_valid)
        except:
            nmi = -1

        try:
            ari = adjusted_rand_score(labels_true_valid, labels_pred_valid)
        except:
            ari = -1

        try:
            v_measure = v_measure_score(labels_true_valid, labels_pred_valid)
        except:
            v_measure = -1

        try:
            cluster_acc = self.calculate_cluster_accuracy(
                labels_true_valid, labels_pred_valid)
        except:
            cluster_acc = -1

        metrics = {
            'Algorithm': algorithm_name,
            'N_Clusters': n_clusters,
            'Silhouette': round(silhouette, 4),
            'Davies_Bouldin': round(davies_bouldin, 4),
            'Calinski_Harabasz': round(calinski_harabasz, 2),
            'NMI': round(nmi, 4),
            'ARI': round(ari, 4),
            'V_Measure': round(v_measure, 4),
            'Cluster_Accuracy': round(cluster_acc, 4),
            'Valid_Samples': len(labels_pred_valid),
            'Noise_Points': np.sum(labels_pred == -1)
        }

        return metrics

    def kmeans_clustering(self, X, y, n_clusters=10, algorithm_name='K-Means'):
        """
        Perform K-Means clustering.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): True labels
            n_clusters (int): Number of clusters
            algorithm_name (str): Name for reporting

        Returns:
            tuple: (model, labels_pred, metrics)
        """
        print(f"\n  Running {algorithm_name}...")

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_pred = model.fit_predict(X)

        metrics = self.evaluate_clustering(X, y, labels_pred, algorithm_name)

        return model, labels_pred, metrics

    def minibatch_kmeans_clustering(self, X, y, n_clusters=10):
        """
        Perform MiniBatch K-Means clustering.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): True labels
            n_clusters (int): Number of clusters

        Returns:
            tuple: (model, labels_pred, metrics)
        """
        print("  Running MiniBatch K-Means...")

        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                                batch_size=100, n_init=10)
        labels_pred = model.fit_predict(X)

        metrics = self.evaluate_clustering(
            X, y, labels_pred, 'MiniBatch K-Means')

        return model, labels_pred, metrics

    def spectral_clustering(self, X, y, n_clusters=10):
        """
        Perform Spectral Clustering.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): True labels
            n_clusters (int): Number of clusters

        Returns:
            tuple: (model, labels_pred, metrics)
        """
        print("  Running Spectral Clustering...")

        model = SpectralClustering(n_clusters=n_clusters, random_state=42,
                                   affinity='nearest_neighbors', n_neighbors=10)
        labels_pred = model.fit_predict(X)

        metrics = self.evaluate_clustering(
            X, y, labels_pred, 'Spectral Clustering')

        return model, labels_pred, metrics

    def dbscan_clustering(self, X, y, eps=2.5, min_samples=5):
        """
        Perform DBSCAN clustering.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): True labels
            eps (float): Maximum distance between samples
            min_samples (int): Minimum samples in neighborhood

        Returns:
            tuple: (model, labels_pred, metrics)
        """
        print("  Running DBSCAN...")

        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels_pred = model.fit_predict(X)

        n_clusters = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
        n_noise = list(labels_pred).count(-1)

        print(f"    → Found {n_clusters} clusters and {n_noise} noise points")

        metrics = self.evaluate_clustering(X, y, labels_pred, 'DBSCAN')

        return model, labels_pred, metrics

    def gmm_clustering(self, X, y, n_components=10):
        """
        Perform Gaussian Mixture Model clustering.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): True labels
            n_components (int): Number of mixture components

        Returns:
            tuple: (model, labels_pred, metrics)
        """
        print("  Running Gaussian Mixture Model...")

        model = GaussianMixture(n_components=n_components, random_state=42,
                                covariance_type='full', n_init=10)
        model.fit(X)
        labels_pred = model.predict(X)

        metrics = self.evaluate_clustering(X, y, labels_pred, 'GMM')

        return model, labels_pred, metrics

    def run_experiment(self, train_test_split_ratio, random_state=42):
        """
        Run clustering experiment with specified train-test split.

        Args:
            train_test_split_ratio (tuple): (train_size, test_size) as percentages
            random_state (int): Random seed

        Returns:
            pd.DataFrame: Results dataframe
        """
        train_size, test_size = train_test_split_ratio
        test_ratio = test_size / 100

        print(f"\n{'=' * 80}")
        print(f"Experiment: {train_size}-{test_size} Split")
        print(f"{'=' * 80}")

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(
            test_size=test_ratio, random_state=random_state
        )

        print(f"  Train samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")

        results = []
        models = {}

        # K-Means
        model, labels, metrics = self.kmeans_clustering(
            X_test, y_test, self.n_genres)
        if metrics:
            results.append(metrics)
            models['K-Means'] = (model, labels)

        # MiniBatch K-Means
        model, labels, metrics = self.minibatch_kmeans_clustering(
            X_test, y_test, self.n_genres)
        if metrics:
            results.append(metrics)
            models['MiniBatch K-Means'] = (model, labels)

        # Spectral Clustering
        model, labels, metrics = self.spectral_clustering(
            X_test, y_test, self.n_genres)
        if metrics:
            results.append(metrics)
            models['Spectral Clustering'] = (model, labels)

        # DBSCAN
        model, labels, metrics = self.dbscan_clustering(X_test, y_test)
        if metrics:
            results.append(metrics)
            models['DBSCAN'] = (model, labels)

        # GMM
        model, labels, metrics = self.gmm_clustering(
            X_test, y_test, self.n_genres)
        if metrics:
            results.append(metrics)
            models['GMM'] = (model, labels)

        results_df = pd.DataFrame(results)
        results_df['Split'] = f"{train_size}-{test_size}"

        return results_df, models, X_test, y_test

    def run_all_experiments(self):
        """
        Run experiments with different train-test splits.

        Returns:
            pd.DataFrame: Combined results from all experiments
        """
        print("\n" + "=" * 80)
        print("RUNNING ALL CLUSTERING EXPERIMENTS")
        print("=" * 80)

        splits = [(50, 50), (60, 40), (70, 30), (80, 20)]
        all_results = []

        for split in splits:
            results_df, models, X_test, y_test = self.run_experiment(split)
            all_results.append(results_df)

        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)

        # Save results
        combined_results.to_csv('results/clustering_results.csv', index=False)
        print(f"\n✓ Results saved: results/clustering_results.csv")

        return combined_results

    def visualize_results(self, results_df):
        """
        Create comprehensive visualizations of clustering results.

        Args:
            results_df (pd.DataFrame): Results dataframe
        """
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)

        import os
        os.makedirs('results', exist_ok=True)

        # 1. Metric Comparison by Algorithm (averaged across splits)
        avg_results = results_df.groupby('Algorithm').mean(numeric_only=True)

        metrics_to_plot = ['Silhouette', 'Davies_Bouldin', 'Calinski_Harabasz',
                           'NMI', 'ARI', 'V_Measure', 'Cluster_Accuracy']

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            avg_results[metric].plot(
                kind='bar', ax=ax, color='skyblue', edgecolor='navy')
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Algorithm', fontweight='bold')
            ax.set_ylabel('Score', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)

        # Remove extra subplot
        fig.delaxes(axes[7])

        plt.tight_layout()
        plt.savefig('results/metrics_comparison.png',
                    dpi=300, bbox_inches='tight')
        print("  ✓ Metrics comparison saved: results/metrics_comparison.png")
        plt.close()

        # 2. Performance by Split
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()

        key_metrics = ['Silhouette', 'NMI', 'ARI', 'V_Measure',
                       'Cluster_Accuracy', 'Davies_Bouldin']

        for idx, metric in enumerate(key_metrics):
            ax = axes[idx]

            for algo in results_df['Algorithm'].unique():
                algo_data = results_df[results_df['Algorithm'] == algo]
                ax.plot(algo_data['Split'], algo_data[metric],
                        marker='o', label=algo, linewidth=2)

            ax.set_title(f'{metric} by Split', fontsize=12, fontweight='bold')
            ax.set_xlabel('Train-Test Split', fontweight='bold')
            ax.set_ylabel(metric, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('results/performance_by_split.png',
                    dpi=300, bbox_inches='tight')
        print("  ✓ Performance by split saved: results/performance_by_split.png")
        plt.close()

        # 3. Radar chart for algorithm comparison
        self._create_radar_chart(avg_results)

    def _create_radar_chart(self, avg_results):
        """
        Create radar chart comparing algorithms.

        Args:
            avg_results (pd.DataFrame): Average results by algorithm
        """
        from math import pi

        # Normalize metrics to 0-1 scale
        metrics = ['Silhouette', 'NMI', 'ARI', 'V_Measure', 'Cluster_Accuracy']

        # For Davies-Bouldin, lower is better, so we invert it
        normalized_data = avg_results[metrics].copy()

        # Normalize to 0-1
        for metric in metrics:
            min_val = normalized_data[metric].min()
            max_val = normalized_data[metric].max()
            if max_val > min_val:
                normalized_data[metric] = (
                    normalized_data[metric] - min_val) / (max_val - min_val)

        # Number of metrics
        num_vars = len(metrics)

        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]

        # Create plot
        fig, ax = plt.subplots(
            figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Plot each algorithm
        colors = plt.cm.Set2(range(len(normalized_data)))

        for idx, (algo, row) in enumerate(normalized_data.iterrows()):
            values = row.values.tolist()
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2,
                    label=algo, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])

        # Fix axis to go in the right order
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.title('Algorithm Performance Comparison\n(Normalized Metrics)',
                  fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig('results/radar_chart.png', dpi=300, bbox_inches='tight')
        print("  ✓ Radar chart saved: results/radar_chart.png")
        plt.close()

    def visualize_clusters_2d(self, X, y_true, y_pred, algorithm_name):
        """
        Visualize clusters in 2D using PCA.

        Args:
            X (np.ndarray): Feature matrix
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            algorithm_name (str): Name of algorithm
        """
        # Further reduce to 2D for visualization
        if X.shape[1] > 2:
            pca_2d = PCA(n_components=2, random_state=42)
            X_2d = pca_2d.fit_transform(X)
        else:
            X_2d = X

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot ground truth
        scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true,
                               cmap='tab10', alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
        ax1.set_title(f'Ground Truth Labels', fontsize=14, fontweight='bold')
        ax1.set_xlabel('First Principal Component', fontweight='bold')
        ax1.set_ylabel('Second Principal Component', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Genre')

        # Plot predictions
        scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred,
                               cmap='tab10', alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
        ax2.set_title(f'{algorithm_name} Predictions',
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('First Principal Component', fontweight='bold')
        ax2.set_ylabel('Second Principal Component', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Cluster')

        plt.tight_layout()
        safe_name = algorithm_name.replace(' ', '_').replace('-', '_').lower()
        plt.savefig(
            f'results/cluster_viz_{safe_name}.png', dpi=300, bbox_inches='tight')
        print(
            f"  ✓ Cluster visualization saved: results/cluster_viz_{safe_name}.png")
        plt.close()

    def generate_summary_table(self, results_df):
        """
        Generate and display summary table.

        Args:
            results_df (pd.DataFrame): Results dataframe
        """
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY TABLE")
        print("=" * 80)

        # Average across all splits
        summary = results_df.groupby('Algorithm').agg({
            'N_Clusters': 'mean',
            'Silhouette': 'mean',
            'Davies_Bouldin': 'mean',
            'Calinski_Harabasz': 'mean',
            'NMI': 'mean',
            'ARI': 'mean',
            'V_Measure': 'mean',
            'Cluster_Accuracy': 'mean'
        }).round(4)

        # Format for display
        summary['N_Clusters'] = summary['N_Clusters'].astype(int)
        summary['Calinski_Harabasz'] = summary['Calinski_Harabasz'].round(2)

        print("\nAverage Performance Across All Splits:")
        print(summary.to_string())

        # Find best algorithm for each metric
        print("\n" + "=" * 80)
        print("BEST ALGORITHM FOR EACH METRIC")
        print("=" * 80)

        metrics = ['Silhouette', 'NMI', 'ARI', 'V_Measure', 'Cluster_Accuracy']
        for metric in metrics:
            best_algo = summary[metric].idxmax()
            best_score = summary[metric].max()
            print(f"  {metric:<25} → {best_algo:<25} ({best_score:.4f})")

        # Davies-Bouldin: lower is better
        best_algo = summary['Davies_Bouldin'].idxmin()
        best_score = summary['Davies_Bouldin'].min()
        print(f"  {'Davies_Bouldin':<25} → {best_algo:<25} ({best_score:.4f})")

        # Calinski-Harabasz: higher is better
        best_algo = summary['Calinski_Harabasz'].idxmax()
        best_score = summary['Calinski_Harabasz'].max()
        print(f"  {'Calinski_Harabasz':<25} → {best_algo:<25} ({best_score:.2f})")

        summary.to_csv('results/summary_table.csv')
        print(f"\n✓ Summary table saved: results/summary_table.csv")

        return summary


if __name__ == "__main__":
    """
    Main execution block for clustering experiments.
    """
    import os
    os.makedirs('results', exist_ok=True)

    # Initialize clusterer
    print("\nInitializing with 30-second features...")
    clusterer = MusicGenreClusterer('gtzan/features_30_sec.csv')

    # Run all experiments
    results = clusterer.run_all_experiments()

    # Generate visualizations
    clusterer.visualize_results(results)

    # Generate summary
    summary = clusterer.generate_summary_table(results)

    # Run one final experiment with best split (80-20) for detailed visualization
    print("\n" + "=" * 80)
    print("Generating Detailed Cluster Visualizations (80-20 split)")
    print("=" * 80)

    results_final, models, X_test, y_test = clusterer.run_experiment((80, 20))

    # Visualize each algorithm
    for algo_name, (model, labels) in models.items():
        clusterer.visualize_clusters_2d(X_test, y_test, labels, algo_name)

    print("\n" + "=" * 80)
    print("ALL CLUSTERING EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print("\nGenerated files in 'results/' directory:")
    print("  - clustering_results.csv")
    print("  - summary_table.csv")
    print("  - metrics_comparison.png")
    print("  - performance_by_split.png")
    print("  - radar_chart.png")
    print("  - cluster_viz_*.png (for each algorithm)")
    print("=" * 80)
