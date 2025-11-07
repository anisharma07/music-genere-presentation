"""
Evaluation metrics for clustering
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score, v_measure_score
)
from typing import Dict, List
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringEvaluator:
    """Evaluate clustering results using multiple metrics"""

    def __init__(self):
        self.results = []

    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray,
                            true_labels: np.ndarray = None,
                            algorithm_name: str = 'Unknown',
                            n_clusters: int = None) -> Dict:
        """
        Evaluate clustering using both internal and external metrics

        Internal metrics (no ground truth needed):
        - Silhouette Score
        - Davies-Bouldin Index
        - Calinski-Harabasz Index

        External metrics (require ground truth):
        - Adjusted Rand Index
        - Normalized Mutual Information
        - V-Measure
        """

        metrics = {
            'algorithm': algorithm_name,
            'n_clusters': n_clusters if n_clusters else len(np.unique(labels[labels != -1]))
        }

        # Filter out noise points (label -1 from DBSCAN)
        valid_mask = labels != -1
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]

        # Need at least 2 clusters for most metrics
        n_unique_labels = len(np.unique(labels_valid))

        if n_unique_labels < 2:
            logger.warning(
                f"{algorithm_name}: Less than 2 clusters found. Skipping metrics.")
            metrics['error'] = 'Less than 2 clusters'
            return metrics

        # Internal metrics
        try:
            metrics['silhouette_score'] = silhouette_score(
                X_valid, labels_valid)
        except Exception as e:
            logger.error(f"Silhouette score error: {str(e)}")
            metrics['silhouette_score'] = np.nan

        try:
            metrics['davies_bouldin_index'] = davies_bouldin_score(
                X_valid, labels_valid)
        except Exception as e:
            logger.error(f"Davies-Bouldin error: {str(e)}")
            metrics['davies_bouldin_index'] = np.nan

        try:
            metrics['calinski_harabasz_index'] = calinski_harabasz_score(
                X_valid, labels_valid)
        except Exception as e:
            logger.error(f"Calinski-Harabasz error: {str(e)}")
            metrics['calinski_harabasz_index'] = np.nan

        # External metrics (if ground truth available)
        if true_labels is not None:
            true_labels_valid = true_labels[valid_mask]

            try:
                metrics['adjusted_rand_index'] = adjusted_rand_score(
                    true_labels_valid, labels_valid)
            except Exception as e:
                logger.error(f"ARI error: {str(e)}")
                metrics['adjusted_rand_index'] = np.nan

            try:
                metrics['normalized_mutual_info'] = normalized_mutual_info_score(
                    true_labels_valid, labels_valid)
            except Exception as e:
                logger.error(f"NMI error: {str(e)}")
                metrics['normalized_mutual_info'] = np.nan

            try:
                metrics['v_measure'] = v_measure_score(
                    true_labels_valid, labels_valid)
            except Exception as e:
                logger.error(f"V-Measure error: {str(e)}")
                metrics['v_measure'] = np.nan

            # Cluster purity
            try:
                metrics['purity'] = self._compute_purity(
                    true_labels_valid, labels_valid)
            except Exception as e:
                logger.error(f"Purity error: {str(e)}")
                metrics['purity'] = np.nan

        # Additional statistics
        metrics['n_noise_points'] = np.sum(labels == -1)
        metrics['noise_ratio'] = np.sum(labels == -1) / len(labels)

        self.results.append(metrics)

        return metrics

    def _compute_purity(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """Compute cluster purity"""

        # Confusion matrix
        contingency_matrix = pd.crosstab(pred_labels, true_labels)

        # Sum of maximum values in each cluster
        purity = np.sum(np.max(contingency_matrix.values,
                        axis=1)) / len(pred_labels)

        return purity

    def evaluate_all_algorithms(self, X: np.ndarray, clustering_results: Dict,
                                true_labels: np.ndarray = None) -> pd.DataFrame:
        """Evaluate all clustering algorithms"""

        logger.info("Evaluating all clustering algorithms...")

        all_metrics = []

        for algo_name, result in clustering_results.items():
            if result is None:
                logger.warning(f"Skipping {algo_name} - no results")
                continue

            labels = result['labels']
            n_clusters = len(np.unique(labels[labels != -1]))

            metrics = self.evaluate_clustering(
                X, labels, true_labels,
                algorithm_name=algo_name,
                n_clusters=n_clusters
            )

            all_metrics.append(metrics)

        # Convert to DataFrame
        df_metrics = pd.DataFrame(all_metrics)

        return df_metrics

    def compare_across_splits(self, results_list: List[Dict]) -> pd.DataFrame:
        """Compare results across different train-test splits"""

        df_all = pd.DataFrame(results_list)

        # Group by algorithm and compute statistics
        summary = df_all.groupby('algorithm').agg({
            'silhouette_score': ['mean', 'std'],
            'davies_bouldin_index': ['mean', 'std'],
            'calinski_harabasz_index': ['mean', 'std'],
        })

        return summary

    def get_best_parameters(self) -> Dict:
        """Get best parameters for each algorithm based on silhouette score"""

        df = pd.DataFrame(self.results)

        best_params = {}

        for algo in df['algorithm'].unique():
            algo_results = df[df['algorithm'] == algo]

            # Sort by silhouette score (higher is better)
            best_idx = algo_results['silhouette_score'].idxmax()
            best_params[algo] = algo_results.loc[best_idx].to_dict()

        return best_params


def main():
    """Example usage"""
    import os
    from config import OUTPUT_DIR, RESULTS_DIR

    # Load clustered data
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'clustered_data.csv'))

    # Load processed features
    metadata_cols = ['track_id', 'artist_name', 'title']
    cluster_cols = [col for col in df.columns if 'cluster' in col]
    feature_cols = [
        col for col in df.columns if col not in metadata_cols + cluster_cols]

    X = df[feature_cols].values

    # Evaluate each algorithm
    evaluator = ClusteringEvaluator()

    clustering_results = {}
    for col in cluster_cols:
        algo_name = col.replace('_cluster', '')
        clustering_results[algo_name] = {
            'labels': df[col].values,
            'model': None
        }

    # Evaluate
    metrics_df = evaluator.evaluate_all_algorithms(X, clustering_results)

    print("\nEvaluation Metrics:")
    print(metrics_df)

    # Save results
    metrics_df.to_csv(os.path.join(
        RESULTS_DIR, 'evaluation_metrics.csv'), index=False)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
