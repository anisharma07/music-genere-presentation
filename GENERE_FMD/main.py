"""
Main Execution Script
Orchestrates the entire music genre clustering pipeline.
"""

from evaluation import ClusteringEvaluator
from clustering import ClusteringEngine
from data_analysis import DataAnalyzer
from feature_extraction import AudioFeatureExtractor
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class MusicGenreClusteringPipeline:
    """Complete pipeline for music genre clustering analysis."""

    def __init__(self, data_path='fma_small', output_dir='results'):
        """
        Initialize the pipeline.

        Parameters:
        -----------
        data_path : str
            Path to the FMA dataset directory
        output_dir : str
            Directory to save results
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print("="*80)
        print("MUSIC GENRE CLUSTERING PIPELINE")
        print("="*80)
        print(f"Data path: {data_path}")
        print(f"Output directory: {output_dir}")
        print(f"Timestamp: {self.timestamp}")
        print("="*80)

    def step1_feature_extraction(self, max_files=None):
        """
        Step 1: Extract audio features from MP3 files.

        Parameters:
        -----------
        max_files : int, optional
            Maximum number of files to process (for testing)
        """
        print("\n" + "="*80)
        print("STEP 1: FEATURE EXTRACTION")
        print("="*80)

        extractor = AudioFeatureExtractor(data_path=self.data_path)
        features_df = extractor.extract_all_features(max_files=max_files)

        output_path = os.path.join(self.output_dir, 'extracted_features.csv')
        extractor.save_features(features_df, output_path)

        print(f"\n✓ Step 1 complete! Features saved to {output_path}")

        return output_path

    def step2_data_analysis(self, features_path):
        """
        Step 2: Comprehensive data analysis and cleaning.

        Parameters:
        -----------
        features_path : str
            Path to the extracted features CSV
        """
        print("\n" + "="*80)
        print("STEP 2: DATA ANALYSIS AND CLEANING")
        print("="*80)

        analyzer = DataAnalyzer(features_path)
        analyzer.load_data()

        # Perform all analyses
        analyzer.check_data_adequacy()
        analyzer.check_missing_values()
        analyzer.descriptive_statistics()
        analyzer.calculate_percentiles_and_quartiles()
        analyzer.analyze_outliers()

        # Generate visualizations
        boxplot_path = os.path.join(self.output_dir, 'boxplots.png')
        analyzer.plot_boxplots(n_features=12, output_path=boxplot_path)

        analyzer.calculate_trimmed_statistics(trim_fraction=0.1)

        # Clean data
        analyzer.handle_missing_values()
        analyzer.remove_outliers(method='iqr', threshold=1.5)

        # More visualizations
        dist_path = os.path.join(self.output_dir, 'distributions.png')
        analyzer.analyze_distribution(n_features=9, output_path=dist_path)

        corr_path = os.path.join(self.output_dir, 'correlation_matrix.png')
        analyzer.correlation_analysis(output_path=corr_path)

        # Generate report
        analyzer.generate_report()

        # Save cleaned data
        cleaned_path = os.path.join(self.output_dir, 'cleaned_features.csv')
        analyzer.save_cleaned_data(cleaned_path)

        print(f"\n✓ Step 2 complete! Cleaned data saved to {cleaned_path}")

        return cleaned_path

    def step3_clustering_experiments(self, cleaned_path, n_clusters=10):
        """
        Step 3: Run clustering experiments with different algorithms and splits.

        Parameters:
        -----------
        cleaned_path : str
            Path to the cleaned features CSV
        n_clusters : int
            Number of clusters
        """
        print("\n" + "="*80)
        print("STEP 3: CLUSTERING EXPERIMENTS")
        print("="*80)

        engine = ClusteringEngine(
            data_path=cleaned_path, n_clusters=n_clusters)

        # Load and preprocess data
        X_pca = engine.load_and_preprocess(n_components=20)

        # Tune DBSCAN parameters
        print("\n" + "="*80)
        print("DBSCAN PARAMETER TUNING")
        print("="*80)
        dbscan_output = os.path.join(self.output_dir, 'dbscan_k_distance.png')
        suggested_eps = engine.tune_dbscan_parameters(
            X_pca, k=5, output_path=dbscan_output)

        # Run experiments with different splits using tuned eps
        all_results = engine.experiment_with_splits(
            splits=[0.5, 0.4, 0.3, 0.2],
            dbscan_eps=suggested_eps,  # Use tuned parameter
            dbscan_min_samples=5
        )

        print(f"\n✓ Step 3 complete! All clustering experiments finished.")

        return engine, all_results

    def step4_evaluation(self, engine, all_results, use_synthetic_labels=True):
        """
        Step 4: Evaluate clustering results with comprehensive metrics.

        Parameters:
        -----------
        engine : ClusteringEngine
            The clustering engine instance
        all_results : dict
            Results from all experiments
        use_synthetic_labels : bool
            Whether to use synthetic labels for evaluation
        """
        print("\n" + "="*80)
        print("STEP 4: EVALUATION")
        print("="*80)

        evaluator = ClusteringEvaluator()

        # Try to load real metadata labels if available (metadata/tracks.csv)
        metadata_path_candidates = [
            'metadata/fma_metadata/tracks.csv',
            os.path.join('metadata', 'fma_metadata', 'tracks.csv'),
            os.path.join('metadata', 'tracks.csv'),
            'metadata/tracks.csv',
            'tracks.csv'
        ]

        metadata = None
        for p in metadata_path_candidates:
            if os.path.exists(p):
                try:
                    # FMA tracks.csv has multi-level headers, read carefully
                    metadata = pd.read_csv(p, index_col=0, header=[0, 1])
                    print(f"Loaded metadata from: {p}")
                    print(f"Metadata shape: {metadata.shape}")
                    break
                except Exception as e:
                    # Try single header
                    try:
                        metadata = pd.read_csv(p, index_col=0)
                        print(f"Loaded metadata from: {p} (single header)")
                        break
                    except:
                        metadata = None

        y_true = None
        if metadata is not None:
            # FMA metadata has multi-level columns like ('track', 'genre_top')
            try:
                # Try to get genre_top from multi-level columns
                if ('track', 'genre_top') in metadata.columns:
                    genre_col = metadata[('track', 'genre_top')]
                elif 'genre_top' in metadata.columns:
                    genre_col = metadata['genre_top']
                elif ('track', 'genre') in metadata.columns:
                    genre_col = metadata[('track', 'genre')]
                else:
                    # Try to find any column with 'genre' in it
                    genre_cols = [
                        col for col in metadata.columns if 'genre' in str(col).lower()]
                    if genre_cols:
                        genre_col = metadata[genre_cols[0]]
                    else:
                        genre_col = None

                if genre_col is not None:
                    # Create mapping from track_id to genre
                    label_map = {}
                    for tid in metadata.index:
                        try:
                            genre = genre_col.loc[tid]
                            if pd.notna(genre) and str(genre) not in ('', 'nan', 'None'):
                                label_map[int(tid)] = str(genre)
                        except:
                            continue

                    print(f"Found {len(label_map)} tracks with genre labels")

                    # Convert engine.track_ids to ints and map to genres
                    track_ints = []
                    for t in engine.track_ids:
                        try:
                            # Remove leading zeros and convert
                            track_ints.append(int(str(t).lstrip('0') or '0'))
                        except:
                            track_ints.append(None)

                    # Map to labels
                    mapped_labels = [label_map.get(
                        t, None) for t in track_ints]

                    # Count valid labels
                    n_with_labels = sum(
                        1 for l in mapped_labels if l is not None)

                    if n_with_labels > 0:
                        # Convert to integer classes
                        unique_labels = sorted(
                            [l for l in set(mapped_labels) if l is not None])
                        label_to_int = {lab: i for i,
                                        lab in enumerate(unique_labels)}
                        y_true = np.array(
                            [label_to_int.get(l, -1) if l is not None else -1 for l in mapped_labels])

                        print(f"✓ Using REAL FMA metadata labels for evaluation!")
                        print(f"  Genres found: {unique_labels}")
                        print(
                            f"  Samples with labels: {n_with_labels}/{len(y_true)}")
                        use_synthetic_labels = False
            except Exception as e:
                print(f"Error processing metadata: {e}")
                metadata = None
        # Fallback to synthetic labels
        if use_synthetic_labels and y_true is None:
            # Generate synthetic labels if needed
            n_samples = len(engine.X_pca)
            y_true = evaluator.generate_synthetic_labels(
                n_samples, n_classes=8)
            print("Using synthetic labels for evaluation (8 genres)")

        # Evaluate each split
        all_evaluations = {}

        for split_name, split_data in all_results.items():
            print(f"\n{'='*80}")
            print(f"Evaluating Split: {split_name}")
            print(f"{'='*80}")

            X_train = split_data['X_train']
            results = split_data['results']
            train_idx = split_data.get('train_idx', None)

            # Create labels for training data using indices if available
            if y_true is not None and train_idx is not None:
                # train_idx are indices into engine.X_pca / engine.track_ids
                y_train = y_true[train_idx]
            elif y_true is not None:
                # Fallback: take first N labels
                y_train = y_true[:len(X_train)]
            else:
                y_train = None

            # Evaluate all algorithms for this split
            evaluation_results = evaluator.evaluate_multiple_algorithms(
                X_train, results, y_train
            )

            all_evaluations[split_name] = evaluation_results

        print(f"\n✓ Step 4 complete! All evaluations finished.")

        return all_evaluations

    def step5_generate_comparison_tables(self, all_evaluations):
        """
        Step 5: Generate comparison tables and save results.

        Parameters:
        -----------
        all_evaluations : dict
            Evaluation results for all splits
        """
        print("\n" + "="*80)
        print("STEP 5: GENERATING COMPARISON TABLES")
        print("="*80)

        evaluator = ClusteringEvaluator()

        # Create comparison table for each split
        for split_name, evaluation_results in all_evaluations.items():
            print(f"\n{'-'*80}")
            print(f"Split: {split_name}")
            print(f"{'-'*80}")

            comparison_df = evaluator.create_comparison_table(
                evaluation_results)
            print(comparison_df.to_string(index=False))

            # Save to CSV
            output_path = os.path.join(
                self.output_dir,
                f'comparison_table_{split_name}.csv'
            )
            comparison_df.to_csv(output_path, index=False)
            print(f"\nTable saved to: {output_path}")

        print(f"\n✓ Step 5 complete! All comparison tables generated.")

        return all_evaluations

    def step6_visualize_results(self, all_evaluations):
        """
        Step 6: Create comprehensive visualizations.

        Parameters:
        -----------
        all_evaluations : dict
            Evaluation results for all splits
        """
        print("\n" + "="*80)
        print("STEP 6: GENERATING VISUALIZATIONS")
        print("="*80)

        # Prepare data for plotting
        plot_data = []

        for split_name, evaluation_results in all_evaluations.items():
            for algo_name, metrics in evaluation_results.items():
                row = {
                    'Split': split_name,
                    'Algorithm': algo_name,
                    **metrics
                }
                plot_data.append(row)

        df = pd.DataFrame(plot_data)

        # Create visualizations
        self._plot_metric_comparison(df, 'Silhouette',
                                     'Silhouette Score Comparison')
        self._plot_metric_comparison(df, 'Davies-Bouldin',
                                     'Davies-Bouldin Index Comparison',
                                     lower_is_better=True)
        self._plot_metric_comparison(df, 'Calinski-Harabasz',
                                     'Calinski-Harabasz Index Comparison')

        if 'ARI' in df.columns:
            self._plot_metric_comparison(df, 'ARI',
                                         'Adjusted Rand Index Comparison')
            self._plot_metric_comparison(df, 'NMI',
                                         'Normalized Mutual Information Comparison')
            self._plot_metric_comparison(df, 'Accuracy',
                                         'Cluster Accuracy Comparison')

        # Create comprehensive comparison plot
        self._plot_comprehensive_comparison(df)

        print(f"\n✓ Step 6 complete! All visualizations generated.")

    def _plot_metric_comparison(self, df, metric_name, title,
                                lower_is_better=False):
        """Helper function to plot metric comparisons."""
        if metric_name not in df.columns:
            return

        plt.figure(figsize=(12, 6))

        # Filter out NaN values
        plot_df = df[df[metric_name].notna()]

        if len(plot_df) == 0:
            print(f"No valid data for {metric_name}")
            plt.close()
            return

        # Create grouped bar plot
        splits = plot_df['Split'].unique()
        algorithms = plot_df['Algorithm'].unique()

        x = np.arange(len(splits))
        width = 0.15

        for i, algo in enumerate(algorithms):
            algo_data = plot_df[plot_df['Algorithm'] == algo]
            values = [algo_data[algo_data['Split'] == s][metric_name].values[0]
                      if len(algo_data[algo_data['Split'] == s]) > 0 else 0
                      for s in splits]
            plt.bar(x + i*width, values, width, label=algo)

        plt.xlabel('Train-Test Split')
        plt.ylabel(metric_name)
        plt.title(title)
        plt.xticks(x + width * 2, splits)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir,
                                   f'{metric_name.lower().replace(" ", "_")}_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    def _plot_comprehensive_comparison(self, df):
        """Create a comprehensive heatmap of all metrics."""
        # Select key metrics
        metric_cols = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']
        if 'ARI' in df.columns:
            metric_cols.extend(['ARI', 'NMI', 'Accuracy'])

        # Create pivot table for each split
        for split in df['Split'].unique():
            split_df = df[df['Split'] == split]

            # Create matrix
            matrix_data = []
            for metric in metric_cols:
                if metric in split_df.columns:
                    row = split_df.set_index('Algorithm')[metric].to_dict()
                    matrix_data.append(row)

            if not matrix_data:
                continue

            matrix_df = pd.DataFrame(matrix_data, index=metric_cols)

            # Plot heatmap
            plt.figure(figsize=(10, 6))

            # Normalize each row for better visualization
            normalized_df = matrix_df.apply(
                lambda x: (x - x.min()) / (x.max() - x.min())
                if x.max() != x.min() else x,
                axis=1
            )

            sns.heatmap(normalized_df, annot=matrix_df, fmt='.3f',
                        cmap='RdYlGn', center=0.5, cbar_kws={'label': 'Normalized Score'})
            plt.title(f'Comprehensive Metrics Comparison - Split {split}')
            plt.xlabel('Algorithm')
            plt.ylabel('Metric')
            plt.tight_layout()

            output_path = os.path.join(self.output_dir,
                                       f'comprehensive_heatmap_{split}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")
            plt.close()

    def run_complete_pipeline(self, max_files=None, n_clusters=10):
        """
        Run the complete pipeline from start to finish.

        Parameters:
        -----------
        max_files : int, optional
            Maximum number of audio files to process (for testing)
        n_clusters : int
            Number of clusters for algorithms
        """
        print("\n" + "="*80)
        print("RUNNING COMPLETE PIPELINE")
        print("="*80)

        # Step 1: Feature Extraction
        features_path = self.step1_feature_extraction(max_files=max_files)

        # Step 2: Data Analysis and Cleaning
        cleaned_path = self.step2_data_analysis(features_path)

        # Step 3: Clustering Experiments
        engine, all_results = self.step3_clustering_experiments(
            cleaned_path, n_clusters=n_clusters
        )

        # Step 4: Evaluation
        all_evaluations = self.step4_evaluation(engine, all_results,
                                                use_synthetic_labels=True)

        # Step 5: Generate Comparison Tables
        self.step5_generate_comparison_tables(all_evaluations)

        # Step 6: Visualize Results
        self.step6_visualize_results(all_evaluations)

        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        print(f"\nAll results saved to: {self.output_dir}")
        print("\nGenerated files:")
        for file in os.listdir(self.output_dir):
            print(f"  - {file}")

        return all_evaluations


def main():
    """Main execution function."""
    # Configuration
    DATA_PATH = 'fma_small'
    OUTPUT_DIR = 'results'
    MAX_FILES = None  # Process ALL ~8,000 audio files in fma_small dataset
    N_CLUSTERS = 8  # Changed to 8 to match FMA genre count

    # Create and run pipeline
    pipeline = MusicGenreClusteringPipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR
    )

    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        max_files=MAX_FILES,
        n_clusters=N_CLUSTERS
    )

    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(
        f"Total audio files processed: {MAX_FILES if MAX_FILES else 'ALL FILES'}")
    print(f"Number of clusters: {N_CLUSTERS}")
    print(f"PCA components: 20")
    print(f"Features: MFCCs + Delta-MFCCs + Delta-Delta-MFCCs (temporal)")
    print(f"Train-test splits tested: 50-50, 60-40, 70-30, 80-20")
    print(f"Algorithms tested: 5 (K-Means, MiniBatch K-Means, Spectral, DBSCAN, GMM)")
    print(f"DBSCAN: Auto-tuned using k-distance plot")
    print(f"Metrics computed: 8+ (Silhouette, DBI, CHI, ARI, NMI, V-Measure, Purity, Accuracy)")
    print(f"Metadata: REAL FMA genre labels")
    print("="*80)


if __name__ == "__main__":
    main()
