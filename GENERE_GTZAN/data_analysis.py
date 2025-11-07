"""
Data Analysis Module for GTZAN Music Genre Dataset
==================================================

This module provides comprehensive data analysis functionality including:
- Descriptive statistical analysis
- Outlier detection and removal
- Data cleaning and preprocessing
- Distribution analysis
- Correlation analysis

Author: Anirudh Sharma
Topic: Unsupervised Music Genre Discovery Using Audio Feature Learning
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import trim_mean
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MusicDataAnalyzer:
    """
    A comprehensive analyzer for music genre dataset.

    This class performs data cleaning, statistical analysis, outlier detection,
    and visualization for audio features extracted from music files.

    Attributes:
        df (pd.DataFrame): The main dataset
        features (list): List of feature column names
        label_col (str): Name of the label column
        cleaned_df (pd.DataFrame): Cleaned version of the dataset
    """

    def __init__(self, filepath):
        """
        Initialize the analyzer with a dataset.

        Args:
            filepath (str): Path to the CSV file containing music features
        """
        print("=" * 80)
        print("Initializing Music Data Analyzer")
        print("=" * 80)

        self.df = pd.read_csv(filepath)
        self.label_col = 'label'

        # Identify feature columns (exclude filename and label)
        self.features = [col for col in self.df.columns
                         if col not in ['filename', 'label']]

        self.cleaned_df = None

        print(f"✓ Dataset loaded successfully")
        print(f"  - Total samples: {len(self.df)}")
        print(f"  - Total features: {len(self.features)}")
        print(f"  - Genres: {self.df[self.label_col].nunique()}")
        print()

    def check_data_adequacy(self):
        """
        Check if the dataset has adequate data for analysis.

        Returns:
            dict: Dictionary containing adequacy metrics
        """
        print("\n" + "=" * 80)
        print("DATA ADEQUACY CHECK")
        print("=" * 80)

        # Basic statistics
        total_samples = len(self.df)
        total_features = len(self.features)
        genres = self.df[self.label_col].unique()

        print(f"\n1. Dataset Size:")
        print(f"   - Total samples: {total_samples}")
        print(f"   - Total features: {total_features}")
        print(f"   - Number of genres: {len(genres)}")

        # Sample to feature ratio
        ratio = total_samples / total_features
        print(f"\n2. Sample-to-Feature Ratio: {ratio:.2f}")
        if ratio > 10:
            print("   ✓ ADEQUATE: Good ratio for machine learning (>10)")
        else:
            print("   ⚠ WARNING: Low ratio, consider dimensionality reduction")

        # Minimum samples per genre for clustering
        min_samples_per_genre = total_samples / len(genres)
        print(f"\n3. Average samples per genre: {min_samples_per_genre:.0f}")
        if min_samples_per_genre >= 50:
            print("   ✓ ADEQUATE: Sufficient samples per genre (≥50)")
        else:
            print("   ⚠ WARNING: Low samples per genre")

        adequacy_report = {
            'total_samples': total_samples,
            'total_features': total_features,
            'num_genres': len(genres),
            'sample_feature_ratio': ratio,
            'avg_samples_per_genre': min_samples_per_genre,
            'is_adequate': ratio > 10 and min_samples_per_genre >= 50
        }

        return adequacy_report

    def check_class_balance(self):
        """
        Check whether the dataset is balanced across different genres.

        Returns:
            pd.DataFrame: Class distribution with statistics
        """
        print("\n" + "=" * 80)
        print("CLASS BALANCE ANALYSIS")
        print("=" * 80)

        # Count samples per genre
        class_counts = self.df[self.label_col].value_counts().sort_index()
        class_percentages = (class_counts / len(self.df) * 100).round(2)

        # Create distribution dataframe
        distribution = pd.DataFrame({
            'Genre': class_counts.index,
            'Count': class_counts.values,
            'Percentage': class_percentages.values
        })

        print("\nClass Distribution:")
        print(distribution.to_string(index=False))

        # Check balance
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count

        print(f"\nBalance Metrics:")
        print(f"  - Maximum samples: {max_count}")
        print(f"  - Minimum samples: {min_count}")
        print(f"  - Imbalance ratio: {imbalance_ratio:.2f}")

        if imbalance_ratio <= 1.5:
            print("  ✓ BALANCED: Dataset is well balanced (ratio ≤ 1.5)")
        elif imbalance_ratio <= 3:
            print("  ⚠ MODERATELY IMBALANCED: Consider balancing techniques")
        else:
            print("  ✗ HIGHLY IMBALANCED: Balancing required")

        # Visualization
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.bar(distribution['Genre'], distribution['Count'],
                color='skyblue', edgecolor='navy')
        plt.xlabel('Genre', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
        plt.title('Class Distribution - Counts',
                  fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        plt.subplot(1, 2, 2)
        colors = plt.cm.Set3(range(len(distribution)))
        plt.pie(distribution['Percentage'], labels=distribution['Genre'],
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Class Distribution - Percentages',
                  fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig('results/class_balance.png', dpi=300, bbox_inches='tight')
        print("\n  → Plot saved: results/class_balance.png")
        plt.close()

        return distribution

    def descriptive_statistics(self):
        """
        Generate comprehensive descriptive statistics for all features.

        Returns:
            pd.DataFrame: Statistical summary of features
        """
        print("\n" + "=" * 80)
        print("DESCRIPTIVE STATISTICAL ANALYSIS")
        print("=" * 80)

        # Basic statistics
        desc_stats = self.df[self.features].describe()

        # Additional statistics
        additional_stats = pd.DataFrame({
            'variance': self.df[self.features].var(),
            'skewness': self.df[self.features].skew(),
            'kurtosis': self.df[self.features].kurtosis(),
            'range': self.df[self.features].max() - self.df[self.features].min(),
            'iqr': self.df[self.features].quantile(0.75) - self.df[self.features].quantile(0.25)
        }).T

        # Combine all statistics
        full_stats = pd.concat([desc_stats, additional_stats])

        print("\nKey Statistics (first 5 features):")
        print(full_stats.iloc[:, :5].to_string())
        print("\n... (showing first 5 features)")

        # Save complete statistics
        full_stats.to_csv('results/descriptive_statistics.csv')
        print("\n  → Full statistics saved: results/descriptive_statistics.csv")

        return full_stats

    def detect_outliers_iqr(self, feature):
        """
        Detect outliers using the Interquartile Range (IQR) method.

        Args:
            feature (str): Name of the feature column

        Returns:
            tuple: (outlier_indices, Q1, Q3, IQR, lower_bound, upper_bound)
        """
        Q1 = self.df[feature].quantile(0.25)
        Q3 = self.df[feature].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = self.df[(self.df[feature] < lower_bound) |
                           (self.df[feature] > upper_bound)].index

        return outliers, Q1, Q3, IQR, lower_bound, upper_bound

    def check_outliers(self, plot_top_n=10):
        """
        Check for outliers in all features using boxplots and IQR method.

        Args:
            plot_top_n (int): Number of features to plot in detail

        Returns:
            dict: Dictionary containing outlier information for each feature
        """
        print("\n" + "=" * 80)
        print("OUTLIER DETECTION")
        print("=" * 80)

        outlier_info = {}

        for feature in self.features:
            outliers, Q1, Q3, IQR, lower, upper = self.detect_outliers_iqr(
                feature)
            outlier_info[feature] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(self.df)) * 100,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower,
                'upper_bound': upper
            }

        # Sort by outlier percentage
        sorted_features = sorted(outlier_info.items(),
                                 key=lambda x: x[1]['percentage'],
                                 reverse=True)

        print("\nTop 10 features with most outliers:")
        print(f"{'Feature':<35} {'Count':<10} {'Percentage':<12}")
        print("-" * 60)
        for feat, info in sorted_features[:10]:
            print(
                f"{feat:<35} {info['count']:<10} {info['percentage']:<12.2f}%")

        # Create boxplots for top features with outliers
        features_to_plot = [feat for feat, _ in sorted_features[:plot_top_n]]

        fig, axes = plt.subplots(5, 2, figsize=(15, 20))
        axes = axes.ravel()

        for idx, feature in enumerate(features_to_plot):
            ax = axes[idx]
            self.df.boxplot(column=feature, ax=ax)
            ax.set_title(f'{feature}\n({outlier_info[feature]["count"]} outliers, '
                         f'{outlier_info[feature]["percentage"]:.1f}%)',
                         fontsize=10, fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/outlier_boxplots.png',
                    dpi=300, bbox_inches='tight')
        print("\n  → Boxplots saved: results/outlier_boxplots.png")
        plt.close()

        return outlier_info

    def handle_missing_values(self):
        """
        Check for and handle missing values in the dataset.

        Returns:
            pd.DataFrame: Summary of missing values
        """
        print("\n" + "=" * 80)
        print("MISSING VALUE ANALYSIS")
        print("=" * 80)

        # Check for missing values
        missing_count = self.df.isnull().sum()
        missing_percentage = (missing_count / len(self.df)) * 100

        missing_df = pd.DataFrame({
            'Feature': missing_count.index,
            'Missing_Count': missing_count.values,
            'Percentage': missing_percentage.values
        })

        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
            'Missing_Count', ascending=False)

        if len(missing_df) > 0:
            print(f"\nFound {len(missing_df)} features with missing values:")
            print(missing_df.to_string(index=False))

            # Fill missing values with mean
            print("\n  → Filling missing values with column means...")
            for feature in missing_df['Feature']:
                if feature != 'label':
                    mean_value = self.df[feature].mean()
                    self.df[feature].fillna(mean_value, inplace=True)

            print("  ✓ Missing values handled")
        else:
            print("\n  ✓ No missing values found in the dataset!")

        return missing_df

    def remove_outliers(self, method='iqr', threshold=1.5):
        """
        Remove outliers from the dataset.

        Args:
            method (str): Method for outlier removal ('iqr' or 'zscore')
            threshold (float): Threshold for IQR method or z-score

        Returns:
            pd.DataFrame: Cleaned dataset without outliers
        """
        print("\n" + "=" * 80)
        print(f"OUTLIER REMOVAL ({method.upper()} method)")
        print("=" * 80)

        initial_size = len(self.df)

        if method == 'iqr':
            # IQR method
            outlier_mask = pd.Series([False] * len(self.df))

            for feature in self.features:
                Q1 = self.df[feature].quantile(0.25)
                Q3 = self.df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR

                feature_outliers = (self.df[feature] < lower) | (
                    self.df[feature] > upper)
                outlier_mask = outlier_mask | feature_outliers

            self.cleaned_df = self.df[~outlier_mask].copy()

        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(self.df[self.features]))
            outlier_mask = (z_scores > threshold).any(axis=1)
            self.cleaned_df = self.df[~outlier_mask].copy()

        final_size = len(self.cleaned_df)
        removed = initial_size - final_size

        print(f"\nOutlier Removal Summary:")
        print(f"  - Initial samples: {initial_size}")
        print(
            f"  - Samples removed: {removed} ({(removed/initial_size)*100:.2f}%)")
        print(f"  - Final samples: {final_size}")
        print(f"  ✓ Cleaned dataset created")

        return self.cleaned_df

    def analyze_distribution(self, save_plots=True):
        """
        Analyze and visualize the distribution pattern of features.

        Args:
            save_plots (bool): Whether to save distribution plots

        Returns:
            dict: Distribution analysis results
        """
        print("\n" + "=" * 80)
        print("DISTRIBUTION PATTERN ANALYSIS")
        print("=" * 80)

        distribution_info = {}

        for feature in self.features:
            # Normality test (Shapiro-Wilk)
            try:
                # Sample if too large
                sample_data = self.df[feature].dropna()
                if len(sample_data) > 5000:
                    sample_data = sample_data.sample(5000, random_state=42)

                statistic, p_value = stats.shapiro(sample_data)
                is_normal = p_value > 0.05
            except:
                is_normal = False
                p_value = 0

            distribution_info[feature] = {
                'mean': self.df[feature].mean(),
                'median': self.df[feature].median(),
                'std': self.df[feature].std(),
                'skewness': self.df[feature].skew(),
                'kurtosis': self.df[feature].kurtosis(),
                'is_normal': is_normal,
                'normality_p_value': p_value
            }

        # Count normal distributions
        normal_count = sum(
            1 for info in distribution_info.values() if info['is_normal'])

        print(f"\nDistribution Summary:")
        print(f"  - Total features: {len(self.features)}")
        print(f"  - Normal distributions: {normal_count}")
        print(
            f"  - Non-normal distributions: {len(self.features) - normal_count}")

        # Plot distribution for selected features
        if save_plots:
            sample_features = self.features[:6]  # Plot first 6 features

            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            axes = axes.ravel()

            for idx, feature in enumerate(sample_features):
                ax = axes[idx]

                # Histogram with KDE
                self.df[feature].hist(
                    bins=50, ax=ax, alpha=0.7, color='skyblue', edgecolor='black')
                ax2 = ax.twinx()
                self.df[feature].plot(kind='kde', ax=ax2,
                                      color='red', linewidth=2)

                ax.set_xlabel('Value', fontweight='bold')
                ax.set_ylabel('Frequency', fontweight='bold')
                ax2.set_ylabel('Density', fontweight='bold', color='red')

                title = f'{feature}\n'
                title += f"Skew: {distribution_info[feature]['skewness']:.2f}, "
                title += f"Kurt: {distribution_info[feature]['kurtosis']:.2f}"
                ax.set_title(title, fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('results/distribution_analysis.png',
                        dpi=300, bbox_inches='tight')
            print("  → Distribution plots saved: results/distribution_analysis.png")
            plt.close()

        return distribution_info

    def calculate_percentiles_and_quartiles(self):
        """
        Calculate percentiles, quartiles, and related statistics.

        Returns:
            pd.DataFrame: Percentile and quartile information
        """
        print("\n" + "=" * 80)
        print("PERCENTILE AND QUARTILE ANALYSIS")
        print("=" * 80)

        stats_dict = {}

        for feature in self.features:
            stats_dict[feature] = {
                'Mean': self.df[feature].mean(),
                'Median (M)': self.df[feature].median(),
                'Q1 (25th percentile)': self.df[feature].quantile(0.25),
                'Q3 (75th percentile)': self.df[feature].quantile(0.75),
                'P75 (75th percentile)': self.df[feature].quantile(0.75),
                'P25 (25th percentile)': self.df[feature].quantile(0.25),
                'IQR': self.df[feature].quantile(0.75) - self.df[feature].quantile(0.25)
            }

        stats_df = pd.DataFrame(stats_dict).T

        print("\nPercentile Statistics (first 5 features):")
        print(stats_df.head().to_string())
        print("\n... (showing first 5 features)")

        stats_df.to_csv('results/percentile_quartile_stats.csv')
        print("\n  → Full statistics saved: results/percentile_quartile_stats.csv")

        return stats_df

    def calculate_trimmed_statistics(self, trim_fraction=0.1):
        """
        Calculate trimmed mean, trimmed median, and trimmed standard deviation.

        Args:
            trim_fraction (float): Fraction to trim from each end (0 to 0.5)

        Returns:
            pd.DataFrame: Trimmed statistics
        """
        print("\n" + "=" * 80)
        print(f"TRIMMED STATISTICS (trim fraction: {trim_fraction})")
        print("=" * 80)

        trimmed_stats = {}

        for feature in self.features:
            data = self.df[feature].dropna().values

            # Trimmed mean
            trimmed_mean_val = trim_mean(data, trim_fraction)

            # Trimmed median (remove outliers based on percentile)
            lower_p = trim_fraction * 100
            upper_p = 100 - (trim_fraction * 100)
            trimmed_data = data[(data >= np.percentile(data, lower_p)) &
                                (data <= np.percentile(data, upper_p))]
            trimmed_median_val = np.median(trimmed_data)

            # Trimmed standard deviation
            trimmed_std = np.std(trimmed_data)

            trimmed_stats[feature] = {
                'Original_Mean': np.mean(data),
                'Trimmed_Mean': trimmed_mean_val,
                'Original_Median': np.median(data),
                'Trimmed_Median': trimmed_median_val,
                'Original_Std': np.std(data),
                'Trimmed_Std': trimmed_std
            }

        trimmed_df = pd.DataFrame(trimmed_stats).T

        print(f"\nTrimmed Statistics (first 5 features):")
        print(trimmed_df.head().to_string())
        print("\n... (showing first 5 features)")

        trimmed_df.to_csv('results/trimmed_statistics.csv')
        print("\n  → Full statistics saved: results/trimmed_statistics.csv")

        return trimmed_df

    def correlation_analysis(self, method='pearson', top_n=20):
        """
        Perform correlation analysis on important parameters.

        Args:
            method (str): Correlation method ('pearson', 'spearman', 'kendall')
            top_n (int): Number of top correlations to display

        Returns:
            pd.DataFrame: Correlation matrix
        """
        print("\n" + "=" * 80)
        print(f"CORRELATION ANALYSIS ({method.upper()} method)")
        print("=" * 80)

        # Calculate correlation matrix
        corr_matrix = self.df[self.features].corr(method=method)

        # Find top correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Feature_1': corr_matrix.columns[i],
                    'Feature_2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })

        corr_df = pd.DataFrame(corr_pairs)
        corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
        corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)

        print(f"\nTop {top_n} Feature Correlations:")
        print(corr_df.head(top_n)[
              ['Feature_1', 'Feature_2', 'Correlation']].to_string(index=False))

        # Save correlation matrix
        corr_matrix.to_csv('results/correlation_matrix.csv')
        print(f"\n  → Full correlation matrix saved: results/correlation_matrix.csv")

        # Create correlation heatmap
        plt.figure(figsize=(20, 16))

        # Select subset of features for better visualization
        n_features = min(30, len(self.features))
        selected_features = self.features[:n_features]
        corr_subset = self.df[selected_features].corr(method=method)

        sns.heatmap(corr_subset, annot=False, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f'Feature Correlation Heatmap ({method.capitalize()})\n(First {n_features} features)',
                  fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('results/correlation_heatmap.png',
                    dpi=300, bbox_inches='tight')
        print("  → Correlation heatmap saved: results/correlation_heatmap.png")
        plt.close()

        return corr_matrix

    def generate_full_report(self):
        """
        Generate a comprehensive data analysis report.

        This method runs all analysis functions and creates a complete report.
        """
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE DATA ANALYSIS REPORT")
        print("=" * 80)

        # Create results directory
        import os
        os.makedirs('results', exist_ok=True)

        # Run all analyses
        adequacy = self.check_data_adequacy()
        balance = self.check_class_balance()
        desc_stats = self.descriptive_statistics()
        missing = self.handle_missing_values()
        outliers = self.check_outliers()
        distribution = self.analyze_distribution()
        percentiles = self.calculate_percentiles_and_quartiles()
        trimmed = self.calculate_trimmed_statistics()
        correlation = self.correlation_analysis()

        # Clean the data
        cleaned = self.remove_outliers()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print("\nGenerated Files:")
        print("  1. results/class_balance.png")
        print("  2. results/descriptive_statistics.csv")
        print("  3. results/outlier_boxplots.png")
        print("  4. results/distribution_analysis.png")
        print("  5. results/percentile_quartile_stats.csv")
        print("  6. results/trimmed_statistics.csv")
        print("  7. results/correlation_matrix.csv")
        print("  8. results/correlation_heatmap.png")
        print("\nCleaned dataset available in: self.cleaned_df")
        print("=" * 80)

        return {
            'adequacy': adequacy,
            'balance': balance,
            'desc_stats': desc_stats,
            'missing': missing,
            'outliers': outliers,
            'distribution': distribution,
            'percentiles': percentiles,
            'trimmed': trimmed,
            'correlation': correlation,
            'cleaned_data': cleaned
        }


if __name__ == "__main__":
    """
    Main execution block for data analysis.
    """
    # Initialize analyzer with 30-second features
    analyzer = MusicDataAnalyzer('gtzan/features_30_sec.csv')

    # Generate comprehensive report
    report = analyzer.generate_full_report()

    # Save cleaned dataset
    if analyzer.cleaned_df is not None:
        analyzer.cleaned_df.to_csv(
            'gtzan/features_30_sec_cleaned.csv', index=False)
        print("\n✓ Cleaned dataset saved: gtzan/features_30_sec_cleaned.csv")
