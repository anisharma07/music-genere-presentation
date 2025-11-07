"""
Data cleaning and preprocessing module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, List, Dict
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Data cleaning and preprocessing for music features"""

    def __init__(self, config: Dict):
        self.config = config
        self.outlier_info = {}
        self.missing_info = {}

    def clean_data(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Main data cleaning pipeline"""

        logger.info("Starting data cleaning...")

        # Make a copy
        df_clean = df.copy()

        # 1. Handle missing values
        df_clean = self.handle_missing_values(df_clean, feature_cols)

        # 2. Remove or handle outliers
        if self.config.get('remove_outliers', True):
            df_clean = self.handle_outliers(df_clean, feature_cols)

        # 3. Remove infinite values
        df_clean = self.remove_infinite_values(df_clean, feature_cols)

        logger.info(f"Data cleaning complete. Shape: {df_clean.shape}")

        return df_clean

    def handle_missing_values(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Handle missing values in the dataset"""

        logger.info("Handling missing values...")

        # Check for missing values
        missing_counts = df[feature_cols].isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100

        self.missing_info = {
            'counts': missing_counts[missing_counts > 0].to_dict(),
            'percentages': missing_pct[missing_pct > 0].to_dict()
        }

        if len(self.missing_info['counts']) > 0:
            logger.info(
                f"Found missing values in {len(self.missing_info['counts'])} columns")

            method = self.config.get('handle_missing', 'mean')

            if method == 'mean':
                df[feature_cols] = df[feature_cols].fillna(
                    df[feature_cols].mean())
            elif method == 'median':
                df[feature_cols] = df[feature_cols].fillna(
                    df[feature_cols].median())
            elif method == 'drop':
                df = df.dropna(subset=feature_cols)

            logger.info(f"Missing values handled using '{method}' method")
        else:
            logger.info("No missing values found")

        return df

    def remove_infinite_values(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Remove infinite values"""

        # Replace inf with NaN
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

        # Fill NaN with column mean
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

        return df

    def handle_outliers(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Detect and handle outliers"""

        logger.info("Detecting and handling outliers...")

        method = self.config.get('outlier_method', 'iqr')

        outlier_counts = {}

        for col in feature_cols:
            if method == 'iqr':
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                multiplier = self.config.get('iqr_multiplier', 1.5)

                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR

                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs(stats.zscore(df[col]))
                threshold = self.config.get('zscore_threshold', 3)
                outliers = z_scores > threshold

            outlier_count = outliers.sum()
            if outlier_count > 0:
                outlier_counts[col] = outlier_count

                # Cap outliers instead of removing
                if method == 'iqr':
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound

        self.outlier_info = outlier_counts

        if outlier_counts:
            logger.info(
                f"Found outliers in {len(outlier_counts)} columns (capped at bounds)")
        else:
            logger.info("No outliers detected")

        return df

    def get_descriptive_statistics(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Generate comprehensive descriptive statistics"""

        logger.info("Generating descriptive statistics...")

        stats_dict = {}

        for col in feature_cols:
            data = df[col].values

            # Remove NaN values for statistics
            data_clean = data[~np.isnan(data)]

            if len(data_clean) == 0:
                continue

            # Basic statistics
            stats_dict[col] = {
                'count': len(data_clean),
                'mean': np.mean(data_clean),
                'std': np.std(data_clean),
                'min': np.min(data_clean),
                'max': np.max(data_clean),
                'median': np.median(data_clean),
                'Q1': np.percentile(data_clean, 25),
                'Q3': np.percentile(data_clean, 75),
                'P75': np.percentile(data_clean, 75),
                'P25': np.percentile(data_clean, 25),
                'IQR': np.percentile(data_clean, 75) - np.percentile(data_clean, 25),
                'skewness': stats.skew(data_clean),
                'kurtosis': stats.kurtosis(data_clean),
            }

            # Trimmed statistics
            trimming_fraction = self.config.get('trimming_fraction', 0.05)
            trimmed_mean = stats.trim_mean(data_clean, trimming_fraction)

            # Trimmed standard deviation
            n = len(data_clean)
            n_trim = int(n * trimming_fraction)
            data_sorted = np.sort(data_clean)
            data_trimmed = data_sorted[n_trim:-
                                       n_trim] if n_trim > 0 else data_sorted
            trimmed_std = np.std(data_trimmed)

            stats_dict[col]['trimmed_mean'] = trimmed_mean
            stats_dict[col]['trimmed_std'] = trimmed_std
            stats_dict[col]['trimmed_median'] = np.median(data_trimmed)

        return stats_dict

    def plot_boxplots(self, df: pd.DataFrame, feature_cols: List[str],
                      output_dir: str, max_features: int = 20):
        """Create boxplots for outlier visualization"""

        logger.info("Creating boxplots...")

        # Select subset of features if too many
        cols_to_plot = feature_cols[:max_features] if len(
            feature_cols) > max_features else feature_cols

        # Create subplots
        n_cols = 4
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx]
            df.boxplot(column=col, ax=ax)
            ax.set_title(col, fontsize=10)
            ax.tick_params(axis='x', rotation=45)

        # Hide empty subplots
        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'boxplots.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Boxplots saved to {output_dir}")

    def plot_distributions(self, df: pd.DataFrame, feature_cols: List[str],
                           output_dir: str, max_features: int = 20):
        """Plot distribution of features"""

        logger.info("Creating distribution plots...")

        cols_to_plot = feature_cols[:max_features] if len(
            feature_cols) > max_features else feature_cols

        n_cols = 4
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx]
            df[col].hist(bins=50, ax=ax, edgecolor='black')
            ax.set_title(f'Distribution of {col}', fontsize=10)
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')

        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'distributions.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Distribution plots saved to {output_dir}")

    def correlation_analysis(self, df: pd.DataFrame, feature_cols: List[str],
                             output_dir: str):
        """Perform correlation analysis"""

        logger.info("Performing correlation analysis...")

        # Compute correlation matrix
        corr_matrix = df[feature_cols].corr()

        # Plot correlation heatmap (for subset of features)
        plt.figure(figsize=(20, 18))

        # Select important features or subset
        n_features = min(50, len(feature_cols))
        corr_subset = corr_matrix.iloc[:n_features, :n_features]

        sns.heatmap(corr_subset, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Find highly correlated pairs
        threshold = 0.8
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })

        logger.info(
            f"Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > {threshold})")

        return corr_matrix, high_corr_pairs


def main():
    """Example usage"""
    from config import DATA_CLEANING, OUTPUT_DIR, PLOTS_DIR
    import pandas as pd

    # Load data
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'extracted_features.csv'))

    # Get feature columns
    metadata_cols = ['track_id', 'artist_name', 'title']
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    # Clean data
    cleaner = DataCleaner(DATA_CLEANING)
    df_clean = cleaner.clean_data(df, feature_cols)

    # Get statistics
    stats_dict = cleaner.get_descriptive_statistics(df_clean, feature_cols)

    # Save statistics
    stats_df = pd.DataFrame(stats_dict).T
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'descriptive_statistics.csv'))

    # Create visualizations
    cleaner.plot_boxplots(df_clean, feature_cols, PLOTS_DIR)
    cleaner.plot_distributions(df_clean, feature_cols, PLOTS_DIR)
    corr_matrix, high_corr = cleaner.correlation_analysis(
        df_clean, feature_cols, PLOTS_DIR)

    # Save cleaned data
    df_clean.to_csv(os.path.join(
        OUTPUT_DIR, 'cleaned_features.csv'), index=False)

    print("Data cleaning complete!")


if __name__ == "__main__":
    main()
