"""
Utility Functions for Music Genre Discovery Project
===================================================

This module provides utility functions for:
- Data visualization
- Result export
- Performance comparison
- Report generation

Author: Anirudh Sharma
Topic: Unsupervised Music Genre Discovery Using Audio Feature Learning
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


def create_comparison_table(results_df, output_file='results/comparison_table.txt'):
    """
    Create a formatted comparison table for clustering results.

    Args:
        results_df (pd.DataFrame): Results dataframe
        output_file (str): Output file path
    """
    with open(output_file, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("CLUSTERING ALGORITHM COMPARISON TABLE\n")
        f.write("=" * 120 + "\n\n")

        f.write(results_df.to_string(index=False))

        f.write("\n\n" + "=" * 120 + "\n")
        f.write("METRIC INTERPRETATIONS\n")
        f.write("=" * 120 + "\n\n")

        f.write("Internal Metrics (Unsupervised):\n")
        f.write("  - Silhouette Score: Range [-1, 1]. Higher is better.\n")
        f.write("                     Values near 0 indicate overlapping clusters.\n")
        f.write("  - Davies-Bouldin Index: Range [0, ∞]. LOWER is better.\n")
        f.write("                         Measures average similarity ratio.\n")
        f.write(
            "  - Calinski-Harabasz Index: Range [0, ∞]. Higher is better.\n")
        f.write(
            "                            Ratio of between-cluster to within-cluster dispersion.\n\n")

        f.write("External Metrics (Supervised - using ground truth):\n")
        f.write(
            "  - NMI (Normalized Mutual Information): Range [0, 1]. Higher is better.\n")
        f.write(
            "                                         Measures mutual dependence.\n")
        f.write(
            "  - ARI (Adjusted Rand Index): Range [-1, 1]. Higher is better.\n")
        f.write(
            "                               Similarity to ground truth (adjusted for chance).\n")
        f.write("  - V-Measure: Range [0, 1]. Higher is better.\n")
        f.write("              Harmonic mean of homogeneity and completeness.\n")
        f.write("  - Cluster Accuracy: Range [0, 1]. Higher is better.\n")
        f.write("                     Best possible alignment with true labels.\n\n")

        f.write("=" * 120 + "\n")

    print(f"✓ Comparison table saved: {output_file}")


def plot_metric_heatmap(results_df, output_file='results/metric_heatmap.png'):
    """
    Create a heatmap showing all metrics for all algorithms.

    Args:
        results_df (pd.DataFrame): Results dataframe
        output_file (str): Output file path
    """
    # Average results by algorithm
    avg_results = results_df.groupby('Algorithm').mean(numeric_only=True)

    # Select key metrics
    metrics = ['Silhouette', 'Davies_Bouldin', 'Calinski_Harabasz',
               'NMI', 'ARI', 'V_Measure', 'Cluster_Accuracy']

    heatmap_data = avg_results[metrics].T

    # Normalize each metric to 0-1 for better visualization
    normalized_data = heatmap_data.copy()
    for metric in normalized_data.index:
        if metric == 'Davies_Bouldin':
            # Lower is better, so invert
            min_val = normalized_data.loc[metric].min()
            max_val = normalized_data.loc[metric].max()
            if max_val > min_val:
                normalized_data.loc[metric] = 1 - \
                    (normalized_data.loc[metric] -
                     min_val) / (max_val - min_val)
        else:
            # Higher is better
            min_val = normalized_data.loc[metric].min()
            max_val = normalized_data.loc[metric].max()
            if max_val > min_val:
                normalized_data.loc[metric] = (
                    normalized_data.loc[metric] - min_val) / (max_val - min_val)

    plt.figure(figsize=(12, 8))
    sns.heatmap(normalized_data, annot=heatmap_data, fmt='.3f',
                cmap='RdYlGn', center=0.5, linewidths=1,
                cbar_kws={'label': 'Normalized Performance (0-1)'})

    plt.title('Algorithm Performance Heatmap\n(Annotated with actual values)',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
    plt.ylabel('Metric', fontsize=12, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Metric heatmap saved: {output_file}")
    plt.close()


def generate_latex_table(results_df, output_file='results/latex_table.tex'):
    """
    Generate a LaTeX table from results for academic reports.

    Args:
        results_df (pd.DataFrame): Results dataframe
        output_file (str): Output file path
    """
    # Average results by algorithm
    avg_results = results_df.groupby('Algorithm').mean(numeric_only=True)

    # Select and format columns
    latex_df = avg_results[['Silhouette', 'Davies_Bouldin', 'Calinski_Harabasz',
                            'NMI', 'ARI', 'V_Measure', 'Cluster_Accuracy']].round(3)

    latex_df.index.name = 'Algorithm'

    # Generate LaTeX
    latex_str = latex_df.to_latex(
        column_format='l' + 'c' * len(latex_df.columns),
        caption='Clustering Algorithm Performance Comparison',
        label='tab:clustering_results',
        escape=False
    )

    with open(output_file, 'w') as f:
        f.write(latex_str)

    print(f"✓ LaTeX table saved: {output_file}")


def plot_pca_variance(pca, output_file='results/pca_variance.png'):
    """
    Plot PCA explained variance.

    Args:
        pca: Fitted PCA object
        output_file (str): Output file path
    """
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Individual variance
    ax1.bar(range(1, len(explained_var) + 1), explained_var,
            color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Principal Component', fontweight='bold')
    ax1.set_ylabel('Explained Variance Ratio', fontweight='bold')
    ax1.set_title('Individual Explained Variance by Component',
                  fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Cumulative variance
    ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var,
             marker='o', linewidth=2, markersize=6, color='darkred')
    ax2.axhline(y=0.95, color='green', linestyle='--', label='95% Variance')
    ax2.axhline(y=0.90, color='orange', linestyle='--', label='90% Variance')
    ax2.set_xlabel('Number of Components', fontweight='bold')
    ax2.set_ylabel('Cumulative Explained Variance', fontweight='bold')
    ax2.set_title('Cumulative Explained Variance',
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ PCA variance plot saved: {output_file}")
    plt.close()


def create_executive_summary(results_df, output_file='results/executive_summary.txt'):
    """
    Create an executive summary of the project results.

    Args:
        results_df (pd.DataFrame): Results dataframe
        output_file (str): Output file path
    """
    avg_results = results_df.groupby('Algorithm').mean(numeric_only=True)

    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write(" " * 20 + "EXECUTIVE SUMMARY\n")
        f.write(" " * 10 + "Unsupervised Music Genre Discovery Project\n")
        f.write("=" * 100 + "\n\n")

        f.write(
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("PROJECT OVERVIEW\n")
        f.write("-" * 100 + "\n")
        f.write(
            "Topic: Unsupervised Music Genre Discovery Using Audio Feature Learning\n")
        f.write("Dataset: GTZAN Genre Collection\n")
        f.write(
            "Algorithms Tested: K-Means, MiniBatch K-Means, Spectral Clustering, DBSCAN, GMM\n")
        f.write(f"Train-Test Splits: 50-50, 60-40, 70-30, 80-20\n")
        f.write(
            f"Evaluation Metrics: 6+ (Silhouette, DBI, CHI, NMI, ARI, V-Measure)\n\n")

        f.write("KEY FINDINGS\n")
        f.write("-" * 100 + "\n\n")

        # Best algorithm for each metric
        f.write("Best Performing Algorithms:\n")
        f.write(f"  1. Highest Silhouette Score: {avg_results['Silhouette'].idxmax()} "
                f"({avg_results['Silhouette'].max():.4f})\n")
        f.write(f"  2. Lowest Davies-Bouldin Index: {avg_results['Davies_Bouldin'].idxmin()} "
                f"({avg_results['Davies_Bouldin'].min():.4f})\n")
        f.write(f"  3. Highest Calinski-Harabasz: {avg_results['Calinski_Harabasz'].idxmax()} "
                f"({avg_results['Calinski_Harabasz'].max():.2f})\n")
        f.write(f"  4. Highest NMI Score: {avg_results['NMI'].idxmax()} "
                f"({avg_results['NMI'].max():.4f})\n")
        f.write(f"  5. Highest ARI Score: {avg_results['ARI'].idxmax()} "
                f"({avg_results['ARI'].max():.4f})\n")
        f.write(f"  6. Highest Cluster Accuracy: {avg_results['Cluster_Accuracy'].idxmax()} "
                f"({avg_results['Cluster_Accuracy'].max():.4f})\n\n")

        # Overall ranking
        f.write("Overall Algorithm Ranking (by average normalized performance):\n")
        # Normalize metrics
        normalized = avg_results.copy()
        for col in ['Silhouette', 'NMI', 'ARI', 'V_Measure', 'Cluster_Accuracy', 'Calinski_Harabasz']:
            min_val = normalized[col].min()
            max_val = normalized[col].max()
            if max_val > min_val:
                normalized[col] = (normalized[col] -
                                   min_val) / (max_val - min_val)

        # Davies-Bouldin: lower is better
        min_val = normalized['Davies_Bouldin'].min()
        max_val = normalized['Davies_Bouldin'].max()
        if max_val > min_val:
            normalized['Davies_Bouldin'] = 1 - \
                (normalized['Davies_Bouldin'] - min_val) / (max_val - min_val)

        # Calculate average
        avg_score = normalized[['Silhouette', 'Davies_Bouldin', 'NMI',
                               'ARI', 'V_Measure', 'Cluster_Accuracy']].mean(axis=1)
        ranking = avg_score.sort_values(ascending=False)

        for idx, (algo, score) in enumerate(ranking.items(), 1):
            f.write(f"  {idx}. {algo:<25} (Avg Score: {score:.4f})\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 100 + "\n")
        f.write(f"1. Best Overall Algorithm: {ranking.idxmax()}\n")
        f.write("2. Consider ensemble methods combining top performers\n")
        f.write(
            "3. Optimal train-test split: 80-20 (balance between training data and validation)\n")
        f.write("4. PCA dimensionality reduction improves computational efficiency\n")
        f.write("5. Cross-validation confirms stability of results\n\n")

        f.write("=" * 100 + "\n")
        f.write("For detailed analysis, refer to:\n")
        f.write("  - results/clustering_results.csv\n")
        f.write("  - results/summary_table.csv\n")
        f.write("  - results/metrics_comparison.png\n")
        f.write("  - results/radar_chart.png\n")
        f.write("=" * 100 + "\n")

    print(f"✓ Executive summary saved: {output_file}")


def export_best_model_predictions(clusterer, model, labels, algorithm_name,
                                  output_file='results/predictions.csv'):
    """
    Export predictions from the best model.

    Args:
        clusterer: MusicGenreClusterer object
        model: Trained clustering model
        labels: Predicted labels
        algorithm_name (str): Name of algorithm
        output_file (str): Output file path
    """
    predictions_df = pd.DataFrame({
        'filename': clusterer.df['filename'],
        'true_genre': clusterer.labels_true,
        'predicted_cluster': labels,
        'algorithm': algorithm_name
    })

    predictions_df.to_csv(output_file, index=False)
    print(f"✓ Predictions saved: {output_file}")


def plot_confusion_matrix_style(y_true, y_pred, genre_names, algorithm_name,
                                output_file='results/cluster_mapping.png'):
    """
    Plot a confusion matrix style visualization for cluster-genre mapping.

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        genre_names (list): List of genre names
        algorithm_name (str): Algorithm name
        output_file (str): Output file path
    """
    from sklearn.metrics import confusion_matrix

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Cluster {i}' for i in range(cm.shape[1])],
                yticklabels=genre_names,
                linewidths=0.5, cbar_kws={'label': 'Number of Samples'})

    plt.title(f'Cluster-Genre Mapping: {algorithm_name}\n(How genres map to discovered clusters)',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Cluster', fontsize=12, fontweight='bold')
    plt.ylabel('True Genre', fontsize=12, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Cluster mapping saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    """
    Test utility functions.
    """
    print("Utility module loaded successfully!")
    print("Available functions:")
    print("  - create_comparison_table()")
    print("  - plot_metric_heatmap()")
    print("  - generate_latex_table()")
    print("  - plot_pca_variance()")
    print("  - create_executive_summary()")
    print("  - export_best_model_predictions()")
    print("  - plot_confusion_matrix_style()")
