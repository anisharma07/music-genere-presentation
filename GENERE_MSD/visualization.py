"""
Visualization module for clustering results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringVisualizer:
    """Visualize clustering results"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_metrics_comparison(self, metrics_df: pd.DataFrame,
                                filename: str = 'metrics_comparison.png'):
        """Plot comparison of metrics across algorithms"""

        logger.info("Creating metrics comparison plots...")

        # Select numeric columns
        metric_cols = [col for col in metrics_df.columns
                       if col not in ['algorithm', 'n_clusters', 'error', 'n_noise_points', 'noise_ratio']]

        n_metrics = len(metric_cols)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, metric in enumerate(metric_cols[:6]):  # Plot up to 6 metrics
            ax = axes[idx]

            # Filter out NaN values
            plot_data = metrics_df[['algorithm', metric]].dropna()

            if len(plot_data) > 0:
                plot_data.plot(x='algorithm', y=metric,
                               kind='bar', ax=ax, legend=False)
                ax.set_title(
                    f'{metric.replace("_", " ").title()}', fontsize=12)
                ax.set_xlabel('Algorithm')
                ax.set_ylabel('Score')
                ax.tick_params(axis='x', rotation=45)

                # Add value labels on bars
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.3f', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(metric)

        # Hide unused subplots
        for idx in range(len(metric_cols), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename),
                    dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Metrics comparison saved to {filename}")

    def plot_cluster_distribution(self, labels_dict: dict,
                                  filename: str = 'cluster_distribution.png'):
        """Plot distribution of clusters for each algorithm"""

        logger.info("Creating cluster distribution plots...")

        n_algorithms = len(labels_dict)
        n_cols = 2
        n_rows = (n_algorithms + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 5))
        axes = axes.flatten() if n_algorithms > 1 else [axes]

        for idx, (algo_name, labels) in enumerate(labels_dict.items()):
            ax = axes[idx]

            # Count clusters
            unique, counts = np.unique(labels, return_counts=True)

            # Create bar plot
            ax.bar(unique, counts, edgecolor='black')
            ax.set_title(
                f'{algo_name.upper()} - Cluster Distribution', fontsize=12)
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Number of Samples')
            ax.grid(axis='y', alpha=0.3)

            # Add value labels
            for i, (cluster, count) in enumerate(zip(unique, counts)):
                ax.text(cluster, count, str(count),
                        ha='center', va='bottom', fontsize=9)

        # Hide unused subplots
        for idx in range(n_algorithms, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename),
                    dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Cluster distribution saved to {filename}")

    def plot_tsne_visualization(self, X: np.ndarray, labels_dict: dict,
                                filename: str = 'tsne_visualization.png',
                                n_samples: int = 5000):
        """Visualize clusters using t-SNE dimensionality reduction"""

        logger.info("Creating t-SNE visualizations...")

        # Subsample if dataset is too large
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_subset = X[indices]
            labels_subset = {name: labels[indices]
                             for name, labels in labels_dict.items()}
        else:
            X_subset = X
            labels_subset = labels_dict

        # Compute t-SNE
        logger.info("Computing t-SNE embedding...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_subset)

        # Create subplots
        n_algorithms = len(labels_subset)
        n_cols = 2
        n_rows = (n_algorithms + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 7))
        axes = axes.flatten() if n_algorithms > 1 else [axes]

        for idx, (algo_name, labels) in enumerate(labels_subset.items()):
            ax = axes[idx]

            # Plot each cluster with different color
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

            for cluster_id, color in zip(unique_labels, colors):
                mask = labels == cluster_id

                if cluster_id == -1:
                    # Noise points
                    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                               c='gray', alpha=0.3, s=10, label='Noise')
                else:
                    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                               c=[color], alpha=0.6, s=20, label=f'Cluster {cluster_id}')

            ax.set_title(
                f'{algo_name.upper()} - t-SNE Visualization', fontsize=12)
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')

            # Add legend (limit to 15 items to avoid clutter)
            if n_clusters <= 15:
                ax.legend(loc='best', fontsize=8, markerscale=0.7)

        # Hide unused subplots
        for idx in range(n_algorithms, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename),
                    dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"t-SNE visualization saved to {filename}")

    def plot_silhouette_analysis(self, X: np.ndarray, labels: np.ndarray,
                                 algorithm_name: str):
        """Create silhouette analysis plot"""

        from sklearn.metrics import silhouette_samples

        logger.info(f"Creating silhouette analysis for {algorithm_name}...")

        # Filter noise points
        mask = labels != -1
        X_valid = X[mask]
        labels_valid = labels[mask]

        if len(np.unique(labels_valid)) < 2:
            logger.warning(
                f"Less than 2 clusters for {algorithm_name}. Skipping silhouette plot.")
            return

        # Compute silhouette scores
        silhouette_vals = silhouette_samples(X_valid, labels_valid)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))

        y_lower = 10
        unique_labels = np.unique(labels_valid)

        for cluster_id in unique_labels:
            cluster_silhouette_vals = silhouette_vals[labels_valid == cluster_id]
            cluster_silhouette_vals.sort()

            size_cluster = len(cluster_silhouette_vals)
            y_upper = y_lower + size_cluster

            color = plt.cm.nipy_spectral(
                float(cluster_id) / len(unique_labels))
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0, cluster_silhouette_vals,
                             facecolor=color, edgecolor=color, alpha=0.7)

            # Label clusters
            ax.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster_id))

            y_lower = y_upper + 10

        ax.set_title(
            f'Silhouette Analysis - {algorithm_name.upper()}', fontsize=14)
        ax.set_xlabel('Silhouette Coefficient')
        ax.set_ylabel('Cluster Label')

        # Vertical line for average silhouette score
        avg_score = np.mean(silhouette_vals)
        ax.axvline(x=avg_score, color="red", linestyle="--",
                   label=f'Average Score: {avg_score:.3f}')
        ax.legend()

        plt.tight_layout()
        filename = f'silhouette_{algorithm_name}.png'
        plt.savefig(os.path.join(self.output_dir, filename),
                    dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Silhouette analysis saved to {filename}")

    def create_summary_table(self, metrics_df: pd.DataFrame,
                             filename: str = 'metrics_summary_table.png'):
        """Create a formatted table image of metrics"""

        logger.info("Creating summary table...")

        # Select and format columns
        display_df = metrics_df.copy()

        # Round numeric columns
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns
        display_df[numeric_cols] = display_df[numeric_cols].round(4)

        # Create figure
        fig, ax = plt.subplots(figsize=(16, len(display_df) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')

        # Create table
        table = ax.table(cellText=display_df.values,
                         colLabels=display_df.columns,
                         cellLoc='center',
                         loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(len(display_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(display_df) + 1):
            for j in range(len(display_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        plt.title('Clustering Evaluation Metrics Summary',
                  fontsize=14, weight='bold', pad=20)
        plt.savefig(os.path.join(self.output_dir, filename),
                    dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Summary table saved to {filename}")


def main():
    """Example usage"""
    from config import OUTPUT_DIR, PLOTS_DIR, RESULTS_DIR

    # Load data
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'clustered_data.csv'))
    metrics_df = pd.read_csv(os.path.join(
        RESULTS_DIR, 'evaluation_metrics.csv'))

    # Get feature columns
    metadata_cols = ['track_id', 'artist_name', 'title']
    cluster_cols = [col for col in df.columns if 'cluster' in col]
    feature_cols = [
        col for col in df.columns if col not in metadata_cols + cluster_cols]

    X = df[feature_cols].values

    # Extract labels
    labels_dict = {}
    for col in cluster_cols:
        algo_name = col.replace('_cluster', '')
        labels_dict[algo_name] = df[col].values

    # Create visualizations
    visualizer = ClusteringVisualizer(PLOTS_DIR)

    visualizer.plot_metrics_comparison(metrics_df)
    visualizer.plot_cluster_distribution(labels_dict)
    visualizer.plot_tsne_visualization(X, labels_dict, n_samples=3000)
    visualizer.create_summary_table(metrics_df)

    # Silhouette analysis for each algorithm
    for algo_name, labels in labels_dict.items():
        visualizer.plot_silhouette_analysis(X, labels, algo_name)

    print("Visualization complete!")


if __name__ == "__main__":
    main()
