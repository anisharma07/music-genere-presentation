"""
Evaluation Metrics Module
Implements internal and external clustering evaluation metrics.
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score
)
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')


class ClusteringEvaluator:
    """Comprehensive clustering evaluation class."""

    def __init__(self):
        """Initialize the evaluator."""
        pass

    def purity_score(self, y_true, y_pred):
        """
        Calculate purity score for clustering.

        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted cluster labels

        Returns:
        --------
        float
            Purity score (higher is better)
        """
        # Contingency matrix
        contingency_matrix = np.zeros(
            (len(np.unique(y_true)), len(np.unique(y_pred))))

        for i, true_label in enumerate(np.unique(y_true)):
            for j, pred_label in enumerate(np.unique(y_pred)):
                contingency_matrix[i, j] = np.sum(
                    (y_true == true_label) & (y_pred == pred_label))

        # Purity is the sum of maximum values in each column divided by total
        purity = np.sum(np.max(contingency_matrix, axis=0)) / \
            np.sum(contingency_matrix)

        return purity

    def cluster_accuracy(self, y_true, y_pred):
        """
        Calculate cluster accuracy using Hungarian algorithm.

        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted cluster labels

        Returns:
        --------
        float
            Cluster accuracy (higher is better)
        """
        # Remove noise points (-1) if present in DBSCAN
        mask = y_pred != -1
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]

        if len(y_pred_filtered) == 0:
            return 0.0

        # Build contingency matrix
        n_true = len(np.unique(y_true_filtered))
        n_pred = len(np.unique(y_pred_filtered))

        # Map labels to indices
        true_labels = np.unique(y_true_filtered)
        pred_labels = np.unique(y_pred_filtered)

        true_label_map = {label: idx for idx, label in enumerate(true_labels)}
        pred_label_map = {label: idx for idx, label in enumerate(pred_labels)}

        # Create contingency matrix
        matrix = np.zeros((n_true, n_pred))
        for i in range(len(y_true_filtered)):
            true_idx = true_label_map[y_true_filtered[i]]
            pred_idx = pred_label_map[y_pred_filtered[i]]
            matrix[true_idx, pred_idx] += 1

        # Use Hungarian algorithm to find best matching
        row_ind, col_ind = linear_sum_assignment(-matrix)

        # Calculate accuracy
        accuracy = matrix[row_ind, col_ind].sum() / len(y_true_filtered)

        return accuracy

    def evaluate_internal_metrics(self, X, labels):
        """
        Calculate internal clustering metrics (no ground truth needed).

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        labels : array-like
            Cluster labels

        Returns:
        --------
        dict
            Dictionary containing internal metrics
        """
        metrics = {}

        # Filter out noise points for DBSCAN
        mask = labels != -1
        X_filtered = X[mask]
        labels_filtered = labels[mask]

        # Check if we have enough clusters
        n_clusters = len(np.unique(labels_filtered))

        if n_clusters > 1 and len(X_filtered) > n_clusters:
            # Silhouette Score (range: -1 to 1, higher is better)
            try:
                metrics['Silhouette'] = silhouette_score(
                    X_filtered, labels_filtered)
            except:
                metrics['Silhouette'] = np.nan

            # Davies-Bouldin Index (lower is better)
            try:
                metrics['Davies-Bouldin'] = davies_bouldin_score(
                    X_filtered, labels_filtered)
            except:
                metrics['Davies-Bouldin'] = np.nan

            # Calinski-Harabasz Index (higher is better)
            try:
                metrics['Calinski-Harabasz'] = calinski_harabasz_score(
                    X_filtered, labels_filtered)
            except:
                metrics['Calinski-Harabasz'] = np.nan
        else:
            metrics['Silhouette'] = np.nan
            metrics['Davies-Bouldin'] = np.nan
            metrics['Calinski-Harabasz'] = np.nan

        return metrics

    def evaluate_external_metrics(self, y_true, y_pred):
        """
        Calculate external clustering metrics (requires ground truth).

        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted cluster labels

        Returns:
        --------
        dict
            Dictionary containing external metrics
        """
        metrics = {}

        # Filter out noise points for DBSCAN
        mask = y_pred != -1
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]

        if len(y_pred_filtered) > 0:
            # Adjusted Rand Index (range: -1 to 1, higher is better)
            try:
                metrics['ARI'] = adjusted_rand_score(
                    y_true_filtered, y_pred_filtered)
            except:
                metrics['ARI'] = np.nan

            # Normalized Mutual Information (range: 0 to 1, higher is better)
            try:
                metrics['NMI'] = normalized_mutual_info_score(
                    y_true_filtered, y_pred_filtered)
            except:
                metrics['NMI'] = np.nan

            # V-Measure (range: 0 to 1, higher is better)
            try:
                metrics['V-Measure'] = v_measure_score(
                    y_true_filtered, y_pred_filtered)
            except:
                metrics['V-Measure'] = np.nan

            # Purity Index
            try:
                metrics['Purity'] = self.purity_score(
                    y_true_filtered, y_pred_filtered)
            except:
                metrics['Purity'] = np.nan

            # Cluster Accuracy
            try:
                metrics['Accuracy'] = self.cluster_accuracy(
                    y_true_filtered, y_pred_filtered)
            except:
                metrics['Accuracy'] = np.nan
        else:
            metrics['ARI'] = np.nan
            metrics['NMI'] = np.nan
            metrics['V-Measure'] = np.nan
            metrics['Purity'] = np.nan
            metrics['Accuracy'] = np.nan

        return metrics

    def evaluate_all_metrics(self, X, labels, y_true=None):
        """
        Calculate all metrics (internal and external if ground truth provided).

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        labels : array-like
            Cluster labels
        y_true : array-like, optional
            True labels for external metrics

        Returns:
        --------
        dict
            Dictionary containing all metrics
        """
        all_metrics = {}

        # Internal metrics
        internal = self.evaluate_internal_metrics(X, labels)
        all_metrics.update(internal)

        # External metrics (if ground truth available)
        if y_true is not None:
            external = self.evaluate_external_metrics(y_true, labels)
            all_metrics.update(external)

        # Add cluster count
        n_clusters = len(np.unique(labels[labels != -1]))
        all_metrics['#Clusters'] = n_clusters

        return all_metrics

    def evaluate_multiple_algorithms(self, X, results_dict, y_true=None):
        """
        Evaluate multiple clustering algorithms.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        results_dict : dict
            Dictionary containing results from multiple algorithms
        y_true : array-like, optional
            True labels for external metrics

        Returns:
        --------
        dict
            Dictionary containing evaluation results for all algorithms
        """
        evaluation_results = {}

        print("\n" + "="*70)
        print("EVALUATING CLUSTERING RESULTS")
        print("="*70)

        for algo_name, result in results_dict.items():
            print(f"\nEvaluating {algo_name}...")

            labels = result['labels']
            metrics = self.evaluate_all_metrics(X, labels, y_true)

            evaluation_results[algo_name] = metrics

            # Print metrics
            print(f"  #Clusters: {metrics['#Clusters']}")
            print(f"  Silhouette: {metrics['Silhouette']:.4f}" if not np.isnan(
                metrics['Silhouette']) else "  Silhouette: N/A")
            print(f"  Davies-Bouldin: {metrics['Davies-Bouldin']:.4f}" if not np.isnan(
                metrics['Davies-Bouldin']) else "  Davies-Bouldin: N/A")
            print(f"  Calinski-Harabasz: {metrics['Calinski-Harabasz']:.2f}" if not np.isnan(
                metrics['Calinski-Harabasz']) else "  Calinski-Harabasz: N/A")

            if y_true is not None:
                print(f"  ARI: {metrics['ARI']:.4f}" if not np.isnan(
                    metrics['ARI']) else "  ARI: N/A")
                print(f"  NMI: {metrics['NMI']:.4f}" if not np.isnan(
                    metrics['NMI']) else "  NMI: N/A")
                print(f"  V-Measure: {metrics['V-Measure']:.4f}" if not np.isnan(
                    metrics['V-Measure']) else "  V-Measure: N/A")
                print(f"  Purity: {metrics['Purity']:.4f}" if not np.isnan(
                    metrics['Purity']) else "  Purity: N/A")
                print(f"  Accuracy: {metrics['Accuracy']:.4f}" if not np.isnan(
                    metrics['Accuracy']) else "  Accuracy: N/A")

        print("\n✓ Evaluation complete!")

        return evaluation_results

    def create_comparison_table(self, evaluation_results):
        """
        Create a comparison table of all algorithms.

        Parameters:
        -----------
        evaluation_results : dict
            Dictionary containing evaluation results

        Returns:
        --------
        pd.DataFrame
            Comparison table
        """
        import pandas as pd

        # Prepare data for table
        table_data = []

        for algo_name, metrics in evaluation_results.items():
            row = {'Algorithm': algo_name}
            row.update(metrics)
            table_data.append(row)

        df = pd.DataFrame(table_data)

        # Reorder columns
        col_order = ['Algorithm', '#Clusters', 'Silhouette', 'Davies-Bouldin',
                     'Calinski-Harabasz']

        # Add external metrics if present
        if 'ARI' in df.columns:
            col_order.extend(['ARI', 'NMI', 'V-Measure', 'Purity', 'Accuracy'])

        # Filter to existing columns
        col_order = [col for col in col_order if col in df.columns]
        df = df[col_order]

        return df

    def generate_synthetic_labels(self, n_samples, n_classes=8):
        """
        Generate synthetic ground truth labels for evaluation purposes.

        This is useful when actual labels are not available from the FMA metadata.

        Parameters:
        -----------
        n_samples : int
            Number of samples
        n_classes : int
            Number of classes (genres)

        Returns:
        --------
        np.ndarray
            Synthetic labels
        """
        # Create balanced synthetic labels
        labels = np.repeat(np.arange(n_classes), n_samples // n_classes)

        # Add remaining samples
        remaining = n_samples - len(labels)
        if remaining > 0:
            labels = np.concatenate([labels, np.arange(remaining)])

        # Shuffle
        np.random.shuffle(labels)

        return labels


if __name__ == "__main__":
    # Example usage with synthetic data
    from sklearn.datasets import make_blobs

    # Create synthetic data
    X, y_true = make_blobs(n_samples=500, n_features=20,
                           centers=8, random_state=42)

    # Simulate clustering results
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=8, random_state=42)
    y_pred = kmeans.fit_predict(X)

    # Evaluate
    evaluator = ClusteringEvaluator()

    print("Testing with synthetic data...")
    print("\nInternal metrics (no ground truth needed):")
    internal_metrics = evaluator.evaluate_internal_metrics(X, y_pred)
    for metric, value in internal_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nExternal metrics (with ground truth):")
    external_metrics = evaluator.evaluate_external_metrics(y_true, y_pred)
    for metric, value in external_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nAll metrics:")
    all_metrics = evaluator.evaluate_all_metrics(X, y_pred, y_true)
    for metric, value in all_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
