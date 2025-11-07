"""
Main pipeline for Music Genre Discovery
This script runs the complete workflow from feature extraction to evaluation
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Import custom modules
from config import *
from feature_extractor import MillionSongFeatureExtractor
from data_cleaner import DataCleaner
from clustering import MusicGenreClusterer
from evaluation import ClusteringEvaluator
from visualization import ClusteringVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'pipeline.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MusicGenreDiscoveryPipeline:
    """Complete pipeline for unsupervised music genre discovery"""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.feature_extractor = None
        self.data_cleaner = DataCleaner(DATA_CLEANING)
        self.clusterer = MusicGenreClusterer(CLUSTERING_CONFIG, RANDOM_SEED)
        self.evaluator = ClusteringEvaluator()
        self.visualizer = ClusteringVisualizer(PLOTS_DIR)

        logger.info("="*80)
        logger.info("Music Genre Discovery Pipeline Initialized")
        logger.info("="*80)

    def step1_extract_features(self, max_files: int = None) -> pd.DataFrame:
        """Step 1: Extract features from HDF5 files"""

        logger.info("\n" + "="*80)
        logger.info("STEP 1: FEATURE EXTRACTION")
        logger.info("="*80)

        features_path = os.path.join(OUTPUT_DIR, 'extracted_features.csv')

        # Check if features already extracted
        if os.path.exists(features_path):
            logger.info(f"Loading existing features from {features_path}")
            df = pd.read_csv(features_path)
            logger.info(f"Loaded {len(df)} tracks with {df.shape[1]} features")
            return df

        # Extract features
        self.feature_extractor = MillionSongFeatureExtractor(DATA_DIR)
        df = self.feature_extractor.extract_all_features(max_files=max_files)

        # Save
        df.to_csv(features_path, index=False)
        logger.info(f"Features saved to {features_path}")
        logger.info(f"Extracted features from {len(df)} tracks")
        logger.info(f"Total features: {df.shape[1]}")

        return df

    def step2_data_analysis_and_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 2: Data analysis and cleaning"""

        logger.info("\n" + "="*80)
        logger.info("STEP 2: DATA ANALYSIS AND CLEANING")
        logger.info("="*80)

        # Get feature columns
        metadata_cols = ['track_id', 'artist_name', 'title']
        feature_cols = [col for col in df.columns if col not in metadata_cols]

        logger.info(f"Total features to analyze: {len(feature_cols)}")

        # Descriptive statistics
        logger.info("\nGenerating descriptive statistics...")
        stats_dict = self.data_cleaner.get_descriptive_statistics(
            df, feature_cols)
        stats_df = pd.DataFrame(stats_dict).T
        stats_df.to_csv(os.path.join(
            RESULTS_DIR, 'descriptive_statistics.csv'))

        # Print summary
        logger.info("\nSample Statistics (first 5 features):")
        print(stats_df.head()[['mean', 'std', 'median', 'Q1', 'Q3']])

        # Visualizations
        logger.info("\nCreating visualizations...")
        self.data_cleaner.plot_boxplots(
            df, feature_cols, PLOTS_DIR, max_features=30)
        self.data_cleaner.plot_distributions(
            df, feature_cols, PLOTS_DIR, max_features=30)
        corr_matrix, high_corr = self.data_cleaner.correlation_analysis(
            df, feature_cols, PLOTS_DIR)

        logger.info(f"Found {len(high_corr)} highly correlated feature pairs")

        # Clean data
        logger.info("\nCleaning data...")
        df_clean = self.data_cleaner.clean_data(df, feature_cols)

        # Save cleaned data
        df_clean.to_csv(os.path.join(
            OUTPUT_DIR, 'cleaned_features.csv'), index=False)
        logger.info(f"Cleaned data saved. Final shape: {df_clean.shape}")

        return df_clean

    def step3_clustering(self, df: pd.DataFrame, n_clusters_list: list = [10]) -> dict:
        """Step 3: Apply clustering algorithms"""

        logger.info("\n" + "="*80)
        logger.info("STEP 3: CLUSTERING")
        logger.info("="*80)

        # Get feature columns
        metadata_cols = ['track_id', 'artist_name', 'title']
        feature_cols = [col for col in df.columns if col not in metadata_cols]

        X = df[feature_cols].values
        logger.info(f"Feature matrix shape: {X.shape}")

        # Preprocess features (standardization + PCA)
        logger.info(
            f"\nApplying PCA to reduce to {PCA_COMPONENTS} dimensions...")
        X_processed = self.clusterer.preprocess_features(
            X, n_components=PCA_COMPONENTS)

        all_results = {}

        # Run clustering for different number of clusters
        for n_clusters in n_clusters_list:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running clustering with k={n_clusters}")
            logger.info(f"{'='*60}")

            results = self.clusterer.run_all_algorithms(
                X_processed, n_clusters=n_clusters)
            all_results[n_clusters] = results

            # Add cluster labels to dataframe
            for algo_name, result in results.items():
                if result:
                    df[f'{algo_name}_cluster_k{n_clusters}'] = result['labels']

        # Save clustered data
        df.to_csv(os.path.join(OUTPUT_DIR, 'clustered_data.csv'), index=False)
        logger.info("\nClustered data saved")

        # Save models
        self.clusterer.save_models(MODELS_DIR)

        return all_results, X_processed

    def step4_evaluation(self, X: np.ndarray, df: pd.DataFrame,
                         all_results: dict) -> pd.DataFrame:
        """Step 4: Evaluate clustering results"""

        logger.info("\n" + "="*80)
        logger.info("STEP 4: EVALUATION")
        logger.info("="*80)

        all_metrics = []

        for n_clusters, results in all_results.items():
            logger.info(f"\nEvaluating results for k={n_clusters}")

            metrics_df = self.evaluator.evaluate_all_algorithms(X, results)
            metrics_df['k'] = n_clusters

            all_metrics.append(metrics_df)

            # Print results
            logger.info(f"\nMetrics for k={n_clusters}:")
            print(metrics_df[['algorithm', 'n_clusters', 'silhouette_score',
                              'davies_bouldin_index', 'calinski_harabasz_index']].to_string())

        # Combine all metrics
        final_metrics = pd.concat(all_metrics, ignore_index=True)

        # Save evaluation results
        final_metrics.to_csv(os.path.join(
            RESULTS_DIR, 'evaluation_metrics.csv'), index=False)
        logger.info("\nEvaluation metrics saved")

        return final_metrics

    def step5_visualization(self, X: np.ndarray, df: pd.DataFrame,
                            metrics_df: pd.DataFrame):
        """Step 5: Create visualizations"""

        logger.info("\n" + "="*80)
        logger.info("STEP 5: VISUALIZATION")
        logger.info("="*80)

        # Get cluster columns
        cluster_cols = [col for col in df.columns if 'cluster' in col]

        # Extract labels for each algorithm (use k=10 as default)
        labels_dict = {}
        for col in cluster_cols:
            if 'k10' in col:
                algo_name = col.replace('_cluster_k10', '')
                labels_dict[algo_name] = df[col].values

        if not labels_dict:
            # If no k10, use first available
            for col in cluster_cols[:4]:  # Limit to 4 algorithms
                algo_name = col.split('_cluster')[0]
                labels_dict[algo_name] = df[col].values

        logger.info(
            f"Creating visualizations for {len(labels_dict)} algorithms...")

        # Metrics comparison
        self.visualizer.plot_metrics_comparison(metrics_df)

        # Cluster distribution
        self.visualizer.plot_cluster_distribution(labels_dict)

        # t-SNE visualization
        self.visualizer.plot_tsne_visualization(X, labels_dict, n_samples=3000)

        # Summary table
        self.visualizer.create_summary_table(metrics_df)

        # Silhouette analysis for each algorithm
        for algo_name, labels in labels_dict.items():
            try:
                self.visualizer.plot_silhouette_analysis(X, labels, algo_name)
            except Exception as e:
                logger.warning(
                    f"Silhouette plot for {algo_name} failed: {str(e)}")

        logger.info("All visualizations created successfully")

    def step6_cross_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 6: Cross-validation with different train-test splits"""

        logger.info("\n" + "="*80)
        logger.info("STEP 6: CROSS-VALIDATION")
        logger.info("="*80)

        metadata_cols = ['track_id', 'artist_name', 'title']
        cluster_cols = [col for col in df.columns if 'cluster' in col]
        feature_cols = [
            col for col in df.columns if col not in metadata_cols + cluster_cols]

        X = df[feature_cols].values

        cv_results = []

        for train_ratio, test_ratio in SPLIT_RATIOS:
            logger.info(
                f"\nTrain-Test Split: {int(train_ratio*100)}-{int(test_ratio*100)}")

            # Preprocess
            clusterer = MusicGenreClusterer(CLUSTERING_CONFIG, RANDOM_SEED)
            X_processed = clusterer.preprocess_features(
                X, n_components=PCA_COMPONENTS)

            # Split
            X_train, X_test = clusterer.split_data(X_processed, train_ratio)

            # Train on training set
            results = clusterer.run_all_algorithms(X_train, n_clusters=10)

            # Evaluate on test set
            evaluator = ClusteringEvaluator()

            for algo_name, result in results.items():
                if result is None:
                    continue

                # For algorithms that can predict (not spectral)
                if algo_name in ['kmeans', 'minibatch_kmeans', 'gmm']:
                    test_labels = result['model'].predict(X_test)
                elif algo_name == 'dbscan':
                    # DBSCAN doesn't predict, just evaluate on training
                    test_labels = result['labels'][:len(X_test)]
                else:
                    continue

                metrics = evaluator.evaluate_clustering(
                    X_test, test_labels, algorithm_name=algo_name, n_clusters=10
                )
                metrics['split'] = f"{int(train_ratio*100)}-{int(test_ratio*100)}"
                cv_results.append(metrics)

        cv_df = pd.DataFrame(cv_results)
        cv_df.to_csv(os.path.join(
            RESULTS_DIR, 'cross_validation_results.csv'), index=False)

        logger.info("\nCross-validation complete")
        logger.info("\nCross-validation summary:")
        print(cv_df.groupby(['algorithm', 'split'])['silhouette_score'].mean())

        return cv_df

    def generate_report(self, metrics_df: pd.DataFrame, cv_df: pd.DataFrame = None):
        """Generate final report"""

        logger.info("\n" + "="*80)
        logger.info("GENERATING FINAL REPORT")
        logger.info("="*80)

        report_path = os.path.join(RESULTS_DIR, 'final_report.txt')

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MUSIC GENRE DISCOVERY - FINAL REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("="*80 + "\n")
            f.write("1. EVALUATION METRICS SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(metrics_df.to_string() + "\n\n")

            f.write("="*80 + "\n")
            f.write("2. BEST PERFORMING ALGORITHMS\n")
            f.write("="*80 + "\n\n")

            # Best by Silhouette Score
            best_silhouette = metrics_df.loc[metrics_df['silhouette_score'].idxmax(
            )]
            f.write(f"Best Silhouette Score: {best_silhouette['algorithm']} ")
            f.write(
                f"(k={best_silhouette['n_clusters']}, score={best_silhouette['silhouette_score']:.4f})\n\n")

            # Best by Davies-Bouldin (lower is better)
            best_db = metrics_df.loc[metrics_df['davies_bouldin_index'].idxmin(
            )]
            f.write(f"Best Davies-Bouldin Index: {best_db['algorithm']} ")
            f.write(
                f"(k={best_db['n_clusters']}, score={best_db['davies_bouldin_index']:.4f})\n\n")

            # Best by Calinski-Harabasz (higher is better)
            best_ch = metrics_df.loc[metrics_df['calinski_harabasz_index'].idxmax(
            )]
            f.write(f"Best Calinski-Harabasz Index: {best_ch['algorithm']} ")
            f.write(
                f"(k={best_ch['n_clusters']}, score={best_ch['calinski_harabasz_index']:.4f})\n\n")

            if cv_df is not None:
                f.write("="*80 + "\n")
                f.write("3. CROSS-VALIDATION RESULTS\n")
                f.write("="*80 + "\n\n")
                f.write(cv_df.to_string() + "\n\n")

            f.write("="*80 + "\n")
            f.write("4. FILES GENERATED\n")
            f.write("="*80 + "\n\n")
            f.write(f"Output Directory: {OUTPUT_DIR}\n")
            f.write(f"Results Directory: {RESULTS_DIR}\n")
            f.write(f"Plots Directory: {PLOTS_DIR}\n")
            f.write(f"Models Directory: {MODELS_DIR}\n\n")

            f.write("Data Files:\n")
            f.write("  - extracted_features.csv\n")
            f.write("  - cleaned_features.csv\n")
            f.write("  - clustered_data.csv\n\n")

            f.write("Result Files:\n")
            f.write("  - descriptive_statistics.csv\n")
            f.write("  - evaluation_metrics.csv\n")
            f.write("  - cross_validation_results.csv\n\n")

            f.write("Visualization Files:\n")
            f.write("  - boxplots.png\n")
            f.write("  - distributions.png\n")
            f.write("  - correlation_heatmap.png\n")
            f.write("  - metrics_comparison.png\n")
            f.write("  - cluster_distribution.png\n")
            f.write("  - tsne_visualization.png\n")
            f.write("  - silhouette_*.png\n")
            f.write("  - metrics_summary_table.png\n\n")

            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")

        logger.info(f"Final report saved to {report_path}")

    def run_complete_pipeline(self, max_files: int = None,
                              n_clusters_list: list = [5, 10, 15, 20],
                              run_cv: bool = False):
        """Run the complete pipeline"""

        start_time = datetime.now()
        logger.info(f"\nPipeline started at {start_time}")

        try:
            # Step 1: Feature Extraction
            df = self.step1_extract_features(max_files=max_files)

            # Step 2: Data Analysis and Cleaning
            df_clean = self.step2_data_analysis_and_cleaning(df)

            # Step 3: Clustering
            all_results, X_processed = self.step3_clustering(
                df_clean, n_clusters_list)

            # Step 4: Evaluation
            metrics_df = self.step4_evaluation(
                X_processed, df_clean, all_results)

            # Step 5: Visualization
            self.step5_visualization(X_processed, df_clean, metrics_df)

            # Step 6: Cross-validation (optional)
            cv_df = None
            if run_cv:
                cv_df = self.step6_cross_validation(df_clean)

            # Generate final report
            self.generate_report(metrics_df, cv_df)

            end_time = datetime.now()
            duration = end_time - start_time

            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            logger.info(f"Total time: {duration}")
            logger.info(f"Results saved in: {OUTPUT_DIR}")

            return True

        except Exception as e:
            logger.error(
                f"\nPipeline failed with error: {str(e)}", exc_info=True)
            return False


def main():
    """Main entry point"""

    # Create pipeline
    pipeline = MusicGenreDiscoveryPipeline()

    # Configuration
    # Process 500 files for demonstration (set to None for all ~10,000)
    MAX_FILES = 500
    N_CLUSTERS_LIST = [10]  # List of cluster numbers to try
    RUN_CROSS_VALIDATION = False  # Set to True to run cross-validation

    # Run pipeline
    success = pipeline.run_complete_pipeline(
        max_files=MAX_FILES,
        n_clusters_list=N_CLUSTERS_LIST,
        run_cv=RUN_CROSS_VALIDATION
    )

    if success:
        print("\n" + "="*80)
        print("SUCCESS! Check the output directory for results:")
        print(f"  - Data: {OUTPUT_DIR}")
        print(f"  - Results: {RESULTS_DIR}")
        print(f"  - Plots: {PLOTS_DIR}")
        print(f"  - Models: {MODELS_DIR}")
        print("="*80)
    else:
        print("\nPipeline failed. Check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
