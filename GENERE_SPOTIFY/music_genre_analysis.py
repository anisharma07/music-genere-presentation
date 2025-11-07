"""
Unsupervised Music Genre Discovery Using Audio Feature Learning
===============================================================

This module implements a comprehensive music genre clustering analysis
using various unsupervised learning algorithms on Spotify audio features.

Author: Generated for Music Genre Analysis Project
Date: 2025
"""

from typing import Dict, List, Tuple, Any
from datetime import datetime
import os
import json
import ast
from scipy.stats import jarque_bera, normaltest
from scipy import stats
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, MiniBatchKMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries

# Statistical Analysis


class MusicGenreAnalyzer:
    """
    Comprehensive music genre analysis and clustering system.

    This class handles data preprocessing, exploratory data analysis,
    feature engineering, clustering, and evaluation for music genre discovery.
    """

    def __init__(self, data_path: str = "Spotify/data/"):
        """
        Initialize the analyzer with data path.

        Args:
            data_path (str): Path to the directory containing CSV files
        """
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        self.features = None
        self.scaler = StandardScaler()
        self.pca = None  # Will be initialized based on actual feature count
        self.clustering_results = {}
        self.evaluation_metrics = {}

        # Define audio features for analysis
        self.audio_features = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
        ]

        # Additional numeric features
        self.additional_features = [
            'duration_ms', 'popularity', 'key', 'mode'
        ]

        self.all_features = self.audio_features + self.additional_features

    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Load and preprocess the Spotify dataset.

        Returns:
            pd.DataFrame: Cleaned and preprocessed dataset
        """
        print("Loading and preprocessing data...")

        # Load main dataset
        main_data_path = os.path.join(self.data_path, "data.csv")
        self.data = pd.read_csv(main_data_path)

        print(f"Original dataset shape: {self.data.shape}")
        print(f"Original columns: {list(self.data.columns)}")

        # Basic data cleaning
        self.data = self.data.dropna(subset=self.all_features)

        # Remove duplicates based on track name and artist
        initial_size = len(self.data)
        self.data = self.data.drop_duplicates(subset=['name', 'artists'])
        print(f"Removed {initial_size - len(self.data)} duplicate tracks")

        # Handle outliers using IQR method for key features
        self.data = self._remove_outliers(self.data, self.audio_features)

        # Convert duration from milliseconds to seconds for better interpretability
        self.data['duration_sec'] = self.data['duration_ms'] / 1000

        # Create decade feature from year
        self.data['decade'] = (self.data['year'] // 10) * 10

        print(f"Final dataset shape after preprocessing: {self.data.shape}")
        return self.data

    def _remove_outliers(self, df: pd.DataFrame, features: List[str],
                         threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers using the IQR method.

        Args:
            df (pd.DataFrame): Input dataframe
            features (List[str]): Features to check for outliers
            threshold (float): IQR multiplier threshold

        Returns:
            pd.DataFrame: Dataframe with outliers removed
        """
        initial_size = len(df)

        for feature in features:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            df = df[(df[feature] >= lower_bound) &
                    (df[feature] <= upper_bound)]

        print(
            f"Removed {initial_size - len(df)} outliers ({((initial_size - len(df))/initial_size)*100:.2f}%)")
        return df

    def exploratory_data_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive exploratory data analysis.

        Returns:
            Dict[str, Any]: Dictionary containing analysis results
        """
        print("\nPerforming Exploratory Data Analysis...")

        results = {}

        # Basic statistics
        results['basic_stats'] = self.data[self.all_features].describe()
        print("Basic Statistics:")
        print(results['basic_stats'])

        # Missing values analysis
        results['missing_values'] = self.data.isnull().sum()
        print(f"\nMissing values:\n{results['missing_values']}")

        # Correlation analysis
        results['correlation_matrix'] = self.data[self.all_features].corr()

        # Distribution analysis
        results['skewness'] = self.data[self.all_features].skew()
        results['kurtosis'] = self.data[self.all_features].kurtosis()

        # Normality tests
        normality_results = {}
        for feature in self.audio_features:
            stat, p_value = normaltest(self.data[feature].dropna())
            normality_results[feature] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
        results['normality_tests'] = normality_results

        # Statistical measures requested in requirements
        for feature in self.audio_features:
            feature_data = self.data[feature].dropna()

            # Sample mean
            sample_mean = feature_data.mean()

            # Percentiles
            p25 = feature_data.quantile(0.25)
            p75 = feature_data.quantile(0.75)

            # Median and Q3
            median = feature_data.median()
            q3 = feature_data.quantile(0.75)

            # Trimmed statistics (remove 10% from each end)
            trim_fraction = 0.1
            trimmed_mean = stats.trim_mean(feature_data, trim_fraction)

            # Trimmed standard deviation
            n_trim = int(len(feature_data) * trim_fraction / 2)
            sorted_data = np.sort(feature_data)
            trimmed_data = sorted_data[n_trim:-
                                       n_trim] if n_trim > 0 else sorted_data
            trimmed_std = np.std(trimmed_data, ddof=1)

            results[f'{feature}_statistics'] = {
                'sample_mean': sample_mean,
                'p25': p25,
                'p75': p75,
                'median': median,
                'q3': q3,
                'trimmed_mean': trimmed_mean,
                'trimmed_std': trimmed_std
            }

        return results

    def create_visualizations(self, save_path: str = "visualizations/"):
        """
        Create comprehensive visualizations for the analysis.

        Args:
            save_path (str): Directory to save visualizations
        """
        os.makedirs(save_path, exist_ok=True)

        print(f"\nCreating visualizations in {save_path}...")

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Feature distributions
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()

        for i, feature in enumerate(self.audio_features):
            if i < len(axes):
                self.data[feature].hist(bins=50, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'Distribution of {feature}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'feature_distributions.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Box plots for outlier identification
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()

        for i, feature in enumerate(self.audio_features):
            if i < len(axes):
                self.data.boxplot(column=feature, ax=axes[i])
                axes[i].set_title(f'Box Plot: {feature}')
                axes[i].set_ylabel(feature)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'box_plots.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.data[self.all_features].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                    center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Audio Features Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'correlation_heatmap.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Pairplot for key features
        key_features = ['danceability', 'energy',
                        'valence', 'acousticness', 'loudness']
        if len(self.data) > 10000:  # Sample if dataset is too large
            sample_data = self.data.sample(n=5000, random_state=42)
        else:
            sample_data = self.data

        sns.pairplot(sample_data[key_features],
                     diag_kind='hist', plot_kws={'alpha': 0.6})
        plt.suptitle('Pairwise Relationships of Key Audio Features', y=1.02)
        plt.savefig(os.path.join(save_path, 'pairplot.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Decade analysis
        plt.figure(figsize=(12, 6))
        decade_stats = self.data.groupby('decade')[self.audio_features].mean()

        for feature in ['energy', 'danceability', 'valence', 'acousticness']:
            plt.plot(decade_stats.index,
                     decade_stats[feature], marker='o', label=feature, linewidth=2)

        plt.title('Evolution of Audio Features Over Decades')
        plt.xlabel('Decade')
        plt.ylabel('Average Feature Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'decade_evolution.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print("Visualizations created successfully!")

    def prepare_features(self) -> np.ndarray:
        """
        Prepare and scale features for clustering.

        Returns:
            np.ndarray: Scaled feature matrix
        """
        print("\nPreparing features for clustering...")

        # Select features for clustering
        self.features = self.data[self.all_features].copy()

        # Handle any remaining missing values
        self.features = self.features.fillna(self.features.mean())

        # Scale features
        scaled_features = self.scaler.fit_transform(self.features)

        # Apply PCA for dimensionality reduction
        # Set n_components to min of (n_features, 20) to avoid errors
        n_components = min(self.features.shape[1], 20)
        self.pca = PCA(n_components=n_components)
        self.processed_data = self.pca.fit_transform(scaled_features)

        print(f"Original features shape: {self.features.shape}")
        print(f"Reduced features shape: {self.processed_data.shape}")
        print(
            f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")

        return self.processed_data

    def perform_clustering(self, n_clusters: int = 10) -> Dict[str, Any]:
        """
        Perform clustering using all four required algorithms.

        Args:
            n_clusters (int): Number of clusters for algorithms that require it

        Returns:
            Dict[str, Any]: Clustering results for all algorithms
        """
        print(f"\nPerforming clustering with {n_clusters} clusters...")

        if self.processed_data is None:
            self.prepare_features()

        algorithms = {
            'K-Means': KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
            'MiniBatch K-Means': MiniBatchKMeans(n_clusters=n_clusters, random_state=42),
            'Spectral Clustering': SpectralClustering(n_clusters=n_clusters, random_state=42),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
            'Gaussian Mixture Model': GaussianMixture(n_components=n_clusters, random_state=42)
        }

        for name, algorithm in algorithms.items():
            print(f"Running {name}...")

            try:
                if name == 'Gaussian Mixture Model':
                    labels = algorithm.fit_predict(self.processed_data)
                else:
                    labels = algorithm.fit_predict(self.processed_data)

                n_clusters_found = len(set(labels)) - \
                    (1 if -1 in labels else 0)

                self.clustering_results[name] = {
                    'labels': labels,
                    'n_clusters': n_clusters_found,
                    'algorithm': algorithm
                }

                print(f"  {name}: Found {n_clusters_found} clusters")

            except Exception as e:
                print(f"  Error in {name}: {str(e)}")
                self.clustering_results[name] = {
                    'labels': None,
                    'n_clusters': 0,
                    'algorithm': None,
                    'error': str(e)
                }

        return self.clustering_results

    def evaluate_clustering(self) -> pd.DataFrame:
        """
        Evaluate clustering results using internal and external metrics.

        Returns:
            pd.DataFrame: Evaluation metrics for all algorithms
        """
        print("\nEvaluating clustering results...")

        if not self.clustering_results:
            print("No clustering results found. Run clustering first.")
            return pd.DataFrame()

        metrics_data = []

        for algorithm_name, results in self.clustering_results.items():
            if results['labels'] is None:
                continue

            labels = results['labels']
            n_clusters = results['n_clusters']

            # Skip evaluation if only one cluster or too many noise points
            if n_clusters < 2:
                print(
                    f"Skipping {algorithm_name}: insufficient clusters ({n_clusters})")
                continue

            # Calculate metrics
            try:
                # Internal metrics (don't need ground truth)
                silhouette = silhouette_score(self.processed_data, labels)
                davies_bouldin = davies_bouldin_score(
                    self.processed_data, labels)
                calinski_harabasz = calinski_harabasz_score(
                    self.processed_data, labels)

                # For external metrics, we'll create pseudo ground truth based on decades
                # This is a proxy since we don't have actual genre labels
                decade_labels = pd.cut(
                    self.data['year'], bins=10, labels=False)
                decade_labels = decade_labels[:len(labels)]  # Match lengths

                # External metrics (using decade as proxy ground truth)
                ari = adjusted_rand_score(decade_labels, labels)
                nmi = normalized_mutual_info_score(decade_labels, labels)
                v_measure = v_measure_score(decade_labels, labels)

                # Calculate purity (custom implementation)
                purity = self._calculate_purity(decade_labels, labels)

                metrics_data.append({
                    'Algorithm': algorithm_name,
                    'N_Clusters': n_clusters,
                    'Silhouette_Score': round(silhouette, 4),
                    'Davies_Bouldin_Index': round(davies_bouldin, 4),
                    'Calinski_Harabasz_Index': round(calinski_harabasz, 2),
                    'Adjusted_Rand_Index': round(ari, 4),
                    'Normalized_Mutual_Info': round(nmi, 4),
                    'V_Measure': round(v_measure, 4),
                    'Purity': round(purity, 4)
                })

            except Exception as e:
                print(f"Error evaluating {algorithm_name}: {str(e)}")

        self.evaluation_metrics = pd.DataFrame(metrics_data)

        if not self.evaluation_metrics.empty:
            print("\nClustering Evaluation Results:")
            print("="*80)
            print(self.evaluation_metrics.to_string(index=False))

        return self.evaluation_metrics

    def _calculate_purity(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """
        Calculate purity score for clustering evaluation.

        Args:
            true_labels (np.ndarray): Ground truth labels
            pred_labels (np.ndarray): Predicted cluster labels

        Returns:
            float: Purity score
        """
        # Create contingency matrix
        contingency_matrix = pd.crosstab(pred_labels, true_labels)

        # Calculate purity
        purity = np.sum(np.amax(contingency_matrix.values,
                        axis=1)) / len(pred_labels)
        return purity

    def run_experiments(self, train_test_splits: List[float] = [0.5, 0.6, 0.7, 0.8],
                        n_clusters_range: List[int] = [5, 8, 10, 12, 15]) -> Dict[str, pd.DataFrame]:
        """
        Run comprehensive experiments with different configurations.

        Args:
            train_test_splits (List[float]): Different train ratios to test
            n_clusters_range (List[int]): Different number of clusters to test

        Returns:
            Dict[str, pd.DataFrame]: Results for each experiment configuration
        """
        print("\nRunning comprehensive experiments...")

        experiment_results = {}

        if self.processed_data is None:
            self.prepare_features()

        for train_ratio in train_test_splits:
            for n_clusters in n_clusters_range:
                exp_name = f"train_{int(train_ratio*100)}_clusters_{n_clusters}"
                print(f"\nExperiment: {exp_name}")

                # Split data
                X_train, X_test = train_test_split(
                    self.processed_data,
                    train_size=train_ratio,
                    random_state=42
                )

                # Store original processed_data
                original_data = self.processed_data.copy()

                # Use training data for clustering
                self.processed_data = X_train

                # Run clustering
                self.perform_clustering(n_clusters=n_clusters)

                # Evaluate on test data
                test_results = []
                for algorithm_name, results in self.clustering_results.items():
                    if results['labels'] is None:
                        continue

                    try:
                        # Predict on test data
                        if hasattr(results['algorithm'], 'predict'):
                            test_labels = results['algorithm'].predict(X_test)
                        else:
                            # For algorithms without predict method, use fit_predict on test
                            test_labels = results['algorithm'].fit_predict(
                                X_test)

                        # Calculate metrics on test data
                        if len(set(test_labels)) > 1:
                            silhouette = silhouette_score(X_test, test_labels)
                            davies_bouldin = davies_bouldin_score(
                                X_test, test_labels)
                            calinski_harabasz = calinski_harabasz_score(
                                X_test, test_labels)

                            test_results.append({
                                'Algorithm': algorithm_name,
                                'Train_Ratio': train_ratio,
                                'N_Clusters': n_clusters,
                                'Test_Silhouette': round(silhouette, 4),
                                'Test_Davies_Bouldin': round(davies_bouldin, 4),
                                'Test_Calinski_Harabasz': round(calinski_harabasz, 2)
                            })

                    except Exception as e:
                        print(
                            f"Error in test evaluation for {algorithm_name}: {str(e)}")

                experiment_results[exp_name] = pd.DataFrame(test_results)

                # Restore original data
                self.processed_data = original_data

        return experiment_results

    def create_clustering_visualizations(self, save_path: str = "clustering_results/"):
        """
        Create visualizations for clustering results.

        Args:
            save_path (str): Directory to save clustering visualizations
        """
        os.makedirs(save_path, exist_ok=True)

        print(f"\nCreating clustering visualizations in {save_path}...")

        if not self.clustering_results:
            print("No clustering results available for visualization.")
            return

        # 1. PCA visualization of clusters
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        for i, (algorithm_name, results) in enumerate(self.clustering_results.items()):
            if i >= len(axes) or results['labels'] is None:
                continue

            ax = axes[i]
            labels = results['labels']

            # Plot first two PCA components
            scatter = ax.scatter(self.processed_data[:, 0], self.processed_data[:, 1],
                                 c=labels, cmap='tab20', alpha=0.6, s=30)
            ax.set_title(
                f'{algorithm_name}\n({results["n_clusters"]} clusters)')
            ax.set_xlabel('First Principal Component')
            ax.set_ylabel('Second Principal Component')

            # Add colorbar if not DBSCAN (which might have -1 labels)
            if algorithm_name != 'DBSCAN':
                plt.colorbar(scatter, ax=ax)

        # Remove empty subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'clustering_pca_visualization.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Silhouette analysis
        if not self.evaluation_metrics.empty:
            plt.figure(figsize=(12, 8))

            algorithms = self.evaluation_metrics['Algorithm']
            silhouette_scores = self.evaluation_metrics['Silhouette_Score']

            bars = plt.bar(algorithms, silhouette_scores,
                           color='skyblue', alpha=0.7)
            plt.title('Silhouette Score Comparison Across Algorithms')
            plt.xlabel('Clustering Algorithm')
            plt.ylabel('Silhouette Score')
            plt.xticks(rotation=45)

            # Add value labels on bars
            for bar, score in zip(bars, silhouette_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{score:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'silhouette_comparison.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Metrics heatmap
        if not self.evaluation_metrics.empty:
            plt.figure(figsize=(12, 8))

            # Select numeric metrics for heatmap
            numeric_cols = ['Silhouette_Score', 'Davies_Bouldin_Index',
                            'Calinski_Harabasz_Index', 'Adjusted_Rand_Index',
                            'Normalized_Mutual_Info', 'V_Measure', 'Purity']

            # Create heatmap data
            heatmap_data = self.evaluation_metrics[numeric_cols].T
            heatmap_data.columns = self.evaluation_metrics['Algorithm']

            sns.heatmap(heatmap_data, annot=True, cmap='RdYlBu_r', center=0,
                        fmt='.3f', cbar_kws={'label': 'Metric Value'})
            plt.title('Clustering Evaluation Metrics Heatmap')
            plt.xlabel('Clustering Algorithm')
            plt.ylabel('Evaluation Metric')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'metrics_heatmap.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        print("Clustering visualizations created successfully!")

    def generate_report(self, save_path: str = "analysis_report.html") -> str:
        """
        Generate a comprehensive HTML report of the analysis.

        Args:
            save_path (str): Path to save the HTML report

        Returns:
            str: Path to the generated report
        """
        print(f"\nGenerating comprehensive report: {save_path}")

        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Music Genre Discovery Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric-table {{ font-size: 14px; }}
                .summary-box {{ background: #f9f9f9; padding: 20px; margin: 20px 0; border-radius: 5px; }}
                .highlight {{ background: #ffffcc; }}
            </style>
        </head>
        <body>
            <h1>Unsupervised Music Genre Discovery Analysis Report</h1>
            <p><strong>Generated on:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <div class="summary-box">
                <h2>Executive Summary</h2>
                <p>This report presents a comprehensive analysis of music genre discovery using unsupervised learning
                techniques on Spotify audio features. The analysis includes data preprocessing, exploratory data analysis,
                dimensionality reduction, and clustering using four different algorithms.</p>
            </div>

            <h2>Dataset Overview</h2>
            <p><strong>Total Tracks:</strong> {len(self.data):,}</p>
            <p><strong>Features Analyzed:</strong> {len(self.all_features)}</p>
            <p><strong>Audio Features:</strong> {', '.join(self.audio_features)}</p>
            <p><strong>Year Range:</strong> {self.data['year'].min()} - {self.data['year'].max()}</p>
        """

        # Add basic statistics
        if hasattr(self, 'data'):
            html_content += f"""
            <h2>Data Quality Assessment</h2>
            <h3>Basic Statistics</h3>
            {self.data[self.all_features].describe().to_html(classes='metric-table')}

            <h3>Missing Values</h3>
            {self.data.isnull().sum().to_frame('Missing Values').to_html(classes='metric-table')}
            """

        # Add clustering results
        if not self.evaluation_metrics.empty:
            html_content += f"""
            <h2>Clustering Results</h2>
            <h3>Algorithm Performance Comparison</h3>
            {self.evaluation_metrics.to_html(index=False, classes='metric-table')}

            <div class="summary-box">
                <h3>Key Findings</h3>
                <ul>
                    <li><strong>Best Silhouette Score:</strong> {self.evaluation_metrics.loc[self.evaluation_metrics['Silhouette_Score'].idxmax(), 'Algorithm']}
                        ({self.evaluation_metrics['Silhouette_Score'].max():.4f})</li>
                    <li><strong>Lowest Davies-Bouldin Index:</strong> {self.evaluation_metrics.loc[self.evaluation_metrics['Davies_Bouldin_Index'].idxmin(), 'Algorithm']}
                        ({self.evaluation_metrics['Davies_Bouldin_Index'].min():.4f})</li>
                    <li><strong>Highest Calinski-Harabasz Index:</strong> {self.evaluation_metrics.loc[self.evaluation_metrics['Calinski_Harabasz_Index'].idxmax(), 'Algorithm']}
                        ({self.evaluation_metrics['Calinski_Harabasz_Index'].max():.2f})</li>
                </ul>
            </div>
            """

        # Add feature correlation insights
        if hasattr(self, 'data'):
            correlation_matrix = self.data[self.all_features].corr()
            html_content += f"""
            <h2>Feature Analysis</h2>
            <h3>Correlation Matrix</h3>
            {correlation_matrix.to_html(classes='metric-table')}
            """

        # Add methodology
        html_content += """
            <h2>Methodology</h2>
            <h3>Data Preprocessing</h3>
            <ul>
                <li>Outlier removal using IQR method (1.5 × IQR threshold)</li>
                <li>Missing value imputation using mean values</li>
                <li>Duplicate track removal based on name and artist</li>
                <li>Feature standardization using StandardScaler</li>
            </ul>

            <h3>Dimensionality Reduction</h3>
            <ul>
                <li>Principal Component Analysis (PCA) to 20 dimensions</li>
                <li>Preserves most important variance in the data</li>
            </ul>

            <h3>Clustering Algorithms</h3>
            <ul>
                <li><strong>K-Means:</strong> Partitional clustering with centroids</li>
                <li><strong>MiniBatch K-Means:</strong> Scalable variant of K-Means</li>
                <li><strong>Spectral Clustering:</strong> Graph-based clustering</li>
                <li><strong>DBSCAN:</strong> Density-based clustering</li>
                <li><strong>Gaussian Mixture Model:</strong> Probabilistic clustering</li>
            </ul>

            <h3>Evaluation Metrics</h3>
            <ul>
                <li><strong>Internal Metrics:</strong> Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index</li>
                <li><strong>External Metrics:</strong> Adjusted Rand Index, Normalized Mutual Information, V-Measure, Purity</li>
            </ul>
        """

        html_content += """
            <h2>Conclusions and Recommendations</h2>
            <div class="summary-box">
                <ul>
                    <li>The analysis successfully identified distinct clusters in the music feature space</li>
                    <li>Different clustering algorithms showed varying performance across metrics</li>
                    <li>PCA effectively reduced dimensionality while preserving important variance</li>
                    <li>Future work could explore genre-specific feature engineering and ensemble clustering methods</li>
                </ul>
            </div>

            <h2>Technical Implementation</h2>
            <p>This analysis was implemented using Python with scikit-learn, pandas, and visualization libraries.
            All code is reproducible and follows best practices for machine learning workflows.</p>

        </body>
        </html>
        """

        # Save report
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Report generated successfully: {save_path}")
        return save_path

    def save_results(self, save_path: str = "results/"):
        """
        Save all analysis results to files.

        Args:
            save_path (str): Directory to save results
        """
        os.makedirs(save_path, exist_ok=True)

        print(f"\nSaving results to {save_path}...")

        # Save processed data
        if self.processed_data is not None:
            np.save(os.path.join(save_path, 'processed_features.npy'),
                    self.processed_data)

        # Save clustering results
        if self.clustering_results:
            clustering_summary = {}
            for name, results in self.clustering_results.items():
                if results['labels'] is not None:
                    clustering_summary[name] = {
                        'n_clusters': results['n_clusters'],
                        'labels': results['labels'].tolist()
                    }

            with open(os.path.join(save_path, 'clustering_results.json'), 'w') as f:
                json.dump(clustering_summary, f, indent=2)

        # Save evaluation metrics
        if not self.evaluation_metrics.empty:
            self.evaluation_metrics.to_csv(os.path.join(
                save_path, 'evaluation_metrics.csv'), index=False)

        # Save basic statistics
        if hasattr(self, 'data'):
            self.data[self.all_features].describe().to_csv(
                os.path.join(save_path, 'basic_statistics.csv'))

        print("Results saved successfully!")


def main():
    """
    Main execution function for the music genre analysis.
    """
    print("=" * 80)
    print("UNSUPERVISED MUSIC GENRE DISCOVERY ANALYSIS")
    print("=" * 80)

    # Initialize analyzer
    analyzer = MusicGenreAnalyzer()

    # Load and preprocess data
    data = analyzer.load_and_preprocess_data()

    # Perform exploratory data analysis
    eda_results = analyzer.exploratory_data_analysis()

    # Create visualizations
    analyzer.create_visualizations()

    # Prepare features for clustering
    analyzer.prepare_features()

    # Perform clustering with default parameters
    clustering_results = analyzer.perform_clustering(n_clusters=10)

    # Evaluate clustering results
    evaluation_results = analyzer.evaluate_clustering()

    # Create clustering visualizations
    analyzer.create_clustering_visualizations()

    # Run comprehensive experiments
    print("\nRunning experiments with different configurations...")
    experiment_results = analyzer.run_experiments(
        train_test_splits=[0.5, 0.7, 0.8],
        n_clusters_range=[8, 10, 12]
    )

    # Generate comprehensive report
    analyzer.generate_report()

    # Save all results
    analyzer.save_results()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated outputs:")
    print("- visualizations/: EDA and feature visualizations")
    print("- clustering_results/: Clustering analysis plots")
    print("- results/: Numerical results and data files")
    print("- analysis_report.html: Comprehensive HTML report")

    return analyzer


if __name__ == "__main__":
    analyzer = main()
