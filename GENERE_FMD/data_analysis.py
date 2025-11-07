"""
Data Analysis and Cleaning Module
Performs comprehensive statistical analysis, outlier detection, and data cleaning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DataAnalyzer:
    """Comprehensive data analysis and cleaning class."""

    def __init__(self, data_path='extracted_features.csv'):
        """
        Initialize the data analyzer.

        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing extracted features
        """
        self.data_path = data_path
        self.df = None
        self.df_cleaned = None
        self.outliers_info = {}

    def load_data(self):
        """Load the feature data from CSV."""
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)[:10]}... (showing first 10)")
        return self.df

    def check_data_adequacy(self):
        """Check if the dataset is adequate for analysis."""
        print("\n" + "="*70)
        print("DATA ADEQUACY ANALYSIS")
        print("="*70)

        print(f"\nTotal samples: {len(self.df)}")
        # Exclude track_id
        print(f"Total features: {len(self.df.columns) - 1}")

        # Check for minimum sample size
        min_samples = 100
        if len(self.df) >= min_samples:
            print(f"✓ Dataset has sufficient samples (>= {min_samples})")
        else:
            print(f"✗ Dataset may be insufficient (< {min_samples})")

        # Check feature-to-sample ratio
        n_features = len(self.df.columns) - 1
        ratio = len(self.df) / n_features
        print(f"Sample-to-feature ratio: {ratio:.2f}")

        if ratio >= 10:
            print("✓ Good sample-to-feature ratio (>= 10:1)")
        else:
            print("✗ Low sample-to-feature ratio, consider dimensionality reduction")

    def check_missing_values(self):
        """Check for missing values in the dataset."""
        print("\n" + "="*70)
        print("MISSING VALUES ANALYSIS")
        print("="*70)

        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100

        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })

        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(
            'Missing Count', ascending=False
        )

        if len(missing_df) > 0:
            print("\nColumns with missing values:")
            print(missing_df)
        else:
            print("\n✓ No missing values found in the dataset!")

        return missing_df

    def descriptive_statistics(self):
        """Generate comprehensive descriptive statistics."""
        print("\n" + "="*70)
        print("DESCRIPTIVE STATISTICS")
        print("="*70)

        # Exclude track_id column
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        stats_df = self.df[numeric_cols].describe()
        print("\nBasic Statistics:")
        print(stats_df)

        # Additional statistics
        print("\n" + "-"*70)
        print("Additional Statistics:")
        print("-"*70)

        additional_stats = pd.DataFrame({
            'Skewness': self.df[numeric_cols].skew(),
            'Kurtosis': self.df[numeric_cols].kurtosis(),
            'Variance': self.df[numeric_cols].var()
        })

        print(additional_stats.head(10))

        return stats_df, additional_stats

    def calculate_percentiles_and_quartiles(self):
        """Calculate percentiles, median, and quartiles."""
        print("\n" + "="*70)
        print("PERCENTILES AND QUARTILES")
        print("="*70)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        percentile_stats = {}

        for col in numeric_cols:
            data = self.df[col].dropna()

            percentile_stats[col] = {
                'Mean (X̄)': np.mean(data),
                'Median (M)': np.median(data),
                'Q1 (25th percentile)': np.percentile(data, 25),
                'Q3 (75th percentile)': np.percentile(data, 75),
                'p=0.75': np.percentile(data, 75),
                'p=0.25': np.percentile(data, 25),
                'IQR': np.percentile(data, 75) - np.percentile(data, 25)
            }

        percentile_df = pd.DataFrame(percentile_stats).T
        print("\nPercentile Statistics (first 10 features):")
        print(percentile_df.head(10))

        return percentile_df

    def detect_outliers_iqr(self, column):
        """
        Detect outliers using IQR method.

        Parameters:
        -----------
        column : str
            Column name to check for outliers

        Returns:
        --------
        tuple
            (outlier_mask, lower_bound, upper_bound)
        """
        data = self.df[column].dropna()
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = (self.df[column] < lower_bound) | (
            self.df[column] > upper_bound)

        return outlier_mask, lower_bound, upper_bound

    def analyze_outliers(self):
        """Comprehensive outlier analysis."""
        print("\n" + "="*70)
        print("OUTLIER ANALYSIS (IQR Method)")
        print("="*70)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        outlier_summary = []

        for col in numeric_cols:
            outlier_mask, lower, upper = self.detect_outliers_iqr(col)
            n_outliers = outlier_mask.sum()
            pct_outliers = (n_outliers / len(self.df)) * 100

            outlier_summary.append({
                'Feature': col,
                'Outliers': n_outliers,
                'Percentage': pct_outliers,
                'Lower Bound': lower,
                'Upper Bound': upper
            })

            self.outliers_info[col] = {
                'mask': outlier_mask,
                'count': n_outliers,
                'percentage': pct_outliers
            }

        outlier_df = pd.DataFrame(outlier_summary)
        outlier_df = outlier_df[outlier_df['Outliers'] > 0].sort_values(
            'Outliers', ascending=False
        )

        print("\nFeatures with outliers (top 20):")
        print(outlier_df.head(20))

        return outlier_df

    def plot_boxplots(self, n_features=12, output_path='boxplots.png'):
        """
        Create boxplots for visualizing outliers.

        Parameters:
        -----------
        n_features : int
            Number of features to plot
        output_path : str
            Path to save the plot
        """
        numeric_cols = self.df.select_dtypes(
            include=[np.number]).columns[:n_features]

        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        axes = axes.ravel()

        for idx, col in enumerate(numeric_cols):
            if idx < len(axes):
                axes[idx].boxplot(self.df[col].dropna(), vert=True)
                axes[idx].set_title(col, fontsize=10)
                axes[idx].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nBoxplots saved to {output_path}")
        plt.close()

    def calculate_trimmed_statistics(self, trim_fraction=0.1):
        """
        Calculate trimmed mean, median, and standard deviation.

        Parameters:
        -----------
        trim_fraction : float
            Fraction to trim from each end (e.g., 0.1 = 10%)
        """
        print("\n" + "="*70)
        print(
            f"TRIMMED STATISTICS (trimming {trim_fraction*100}% from each end)")
        print("="*70)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        trimmed_stats = {}

        for col in numeric_cols:
            data = self.df[col].dropna()

            # Trimmed mean
            trimmed_mean = stats.trim_mean(data, trim_fraction)

            # Trimmed median (using percentile range)
            lower_pct = trim_fraction * 100
            upper_pct = 100 - (trim_fraction * 100)
            trimmed_data = data[(data >= np.percentile(data, lower_pct)) &
                                (data <= np.percentile(data, upper_pct))]
            trimmed_median = np.median(trimmed_data)

            # Trimmed standard deviation
            trimmed_std = np.std(trimmed_data)

            trimmed_stats[col] = {
                'Original Mean': np.mean(data),
                'Trimmed Mean (X̄_T)': trimmed_mean,
                'Original Median': np.median(data),
                'Trimmed Median (M̄_T)': trimmed_median,
                'Original Std': np.std(data),
                'Trimmed Std (S_T)': trimmed_std
            }

        trimmed_df = pd.DataFrame(trimmed_stats).T
        print("\nTrimmed Statistics (first 10 features):")
        print(trimmed_df.head(10))

        return trimmed_df

    def handle_missing_values(self):
        """Handle missing values by filling with column mean."""
        print("\n" + "="*70)
        print("HANDLING MISSING VALUES")
        print("="*70)

        self.df_cleaned = self.df.copy()

        numeric_cols = self.df_cleaned.select_dtypes(
            include=[np.number]).columns

        for col in numeric_cols:
            if self.df_cleaned[col].isnull().sum() > 0:
                mean_value = self.df_cleaned[col].mean()
                self.df_cleaned[col].fillna(mean_value, inplace=True)
                print(f"Filled {col} with mean value: {mean_value:.4f}")

        print("\n✓ All missing values handled!")

        return self.df_cleaned

    def remove_outliers(self, method='iqr', threshold=1.5):
        """
        Remove outliers from the dataset.

        Parameters:
        -----------
        method : str
            Method for outlier removal ('iqr' or 'zscore')
        threshold : float
            Threshold for outlier detection
        """
        print("\n" + "="*70)
        print(f"REMOVING OUTLIERS (method: {method})")
        print("="*70)

        if self.df_cleaned is None:
            self.df_cleaned = self.df.copy()

        original_size = len(self.df_cleaned)
        numeric_cols = self.df_cleaned.select_dtypes(
            include=[np.number]).columns

        if method == 'iqr':
            # Use IQR method
            mask = pd.Series([True] * len(self.df_cleaned))

            for col in numeric_cols:
                Q1 = self.df_cleaned[col].quantile(0.25)
                Q3 = self.df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR

                col_mask = (self.df_cleaned[col] >= lower) & (
                    self.df_cleaned[col] <= upper)
                mask = mask & col_mask

            self.df_cleaned = self.df_cleaned[mask]

        elif method == 'zscore':
            # Use Z-score method
            z_scores = np.abs(stats.zscore(self.df_cleaned[numeric_cols]))
            mask = (z_scores < threshold).all(axis=1)
            self.df_cleaned = self.df_cleaned[mask]

        removed = original_size - len(self.df_cleaned)
        print(f"\nOriginal size: {original_size}")
        print(f"Cleaned size: {len(self.df_cleaned)}")
        print(f"Removed: {removed} samples ({removed/original_size*100:.2f}%)")

        return self.df_cleaned

    def analyze_distribution(self, n_features=9, output_path='distributions.png'):
        """
        Analyze and visualize distribution patterns.

        Parameters:
        -----------
        n_features : int
            Number of features to plot
        output_path : str
            Path to save the plot
        """
        print("\n" + "="*70)
        print("DISTRIBUTION ANALYSIS")
        print("="*70)

        data = self.df_cleaned if self.df_cleaned is not None else self.df
        numeric_cols = data.select_dtypes(
            include=[np.number]).columns[:n_features]

        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.ravel()

        for idx, col in enumerate(numeric_cols):
            if idx < len(axes):
                axes[idx].hist(data[col].dropna(), bins=30,
                               edgecolor='black', alpha=0.7)
                axes[idx].set_title(
                    f'{col}\nSkew: {data[col].skew():.2f}', fontsize=10)
                axes[idx].set_xlabel('Value')
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nDistribution plots saved to {output_path}")
        plt.close()

    def correlation_analysis(self, output_path='correlation_matrix.png'):
        """Perform correlation analysis."""
        print("\n" + "="*70)
        print("CORRELATION ANALYSIS")
        print("="*70)

        data = self.df_cleaned if self.df_cleaned is not None else self.df
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr()

        # Find highly correlated features (> 0.8)
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })

        if high_corr:
            print("\nHighly correlated feature pairs (|r| > 0.8):")
            high_corr_df = pd.DataFrame(high_corr)
            print(high_corr_df.head(20))
        else:
            print("\nNo highly correlated feature pairs found (|r| > 0.8)")

        # Plot correlation heatmap (subset for readability)
        n_features = min(30, len(numeric_cols))
        subset_cols = numeric_cols[:n_features]

        plt.figure(figsize=(16, 14))
        sns.heatmap(corr_matrix.loc[subset_cols, subset_cols],
                    annot=False, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5)
        plt.title('Correlation Matrix (Subset of Features)', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nCorrelation heatmap saved to {output_path}")
        plt.close()

        return corr_matrix

    def save_cleaned_data(self, output_path='cleaned_features.csv'):
        """Save cleaned data to CSV."""
        if self.df_cleaned is not None:
            self.df_cleaned.to_csv(output_path, index=False)
            print(f"\n✓ Cleaned data saved to {output_path}")
        else:
            print("\n✗ No cleaned data available. Run cleaning methods first.")

    def generate_report(self):
        """Generate a comprehensive analysis report."""
        print("\n" + "="*70)
        print("POPULATION ANALYSIS SUMMARY")
        print("="*70)

        data = self.df_cleaned if self.df_cleaned is not None else self.df
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        print("\nBased on the statistical analysis, we can infer:")
        print("\n1. Data Distribution:")

        # Check normality
        skewness = data[numeric_cols].skew()
        highly_skewed = skewness[abs(skewness) > 1].count()
        print(
            f"   - {highly_skewed}/{len(numeric_cols)} features are highly skewed (|skew| > 1)")

        print("\n2. Variability:")
        cv = (data[numeric_cols].std() / data[numeric_cols].mean()) * 100
        high_variability = cv[cv > 50].count()
        print(
            f"   - {high_variability}/{len(numeric_cols)} features show high variability (CV > 50%)")

        print("\n3. Outliers:")
        total_outliers = sum([info['count']
                             for info in self.outliers_info.values()])
        print(f"   - Total outlier instances: {total_outliers}")

        print("\n4. Recommendations:")
        print("   - Consider dimensionality reduction (PCA) to 20D as planned")
        print("   - Standardization/normalization recommended before clustering")
        print("   - Multiple clustering algorithms appropriate due to data complexity")


if __name__ == "__main__":
    # Example usage
    analyzer = DataAnalyzer('extracted_features.csv')

    # Load data
    analyzer.load_data()

    # Perform analyses
    analyzer.check_data_adequacy()
    analyzer.check_missing_values()
    analyzer.descriptive_statistics()
    analyzer.calculate_percentiles_and_quartiles()
    analyzer.analyze_outliers()
    analyzer.plot_boxplots(n_features=12, output_path='boxplots.png')
    analyzer.calculate_trimmed_statistics(trim_fraction=0.1)

    # Clean data
    analyzer.handle_missing_values()
    analyzer.remove_outliers(method='iqr', threshold=1.5)

    # Visualize distributions and correlations
    analyzer.analyze_distribution(
        n_features=9, output_path='distributions.png')
    analyzer.correlation_analysis(output_path='correlation_matrix.png')

    # Generate report
    analyzer.generate_report()

    # Save cleaned data
    analyzer.save_cleaned_data('cleaned_features.csv')
