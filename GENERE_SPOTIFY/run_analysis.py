#!/usr/bin/env python3
"""
Simple runner script for the Music Genre Analysis
================================================

This script provides an easy way to run the complete analysis
with different configurations and parameters.

Usage:
    python run_analysis.py [options]

Options:
    --quick: Run quick analysis with reduced dataset size
    --full: Run complete analysis (default)
    --clusters N: Number of clusters to use (default: 10)
    --experiments: Run comprehensive experiments with multiple configurations
"""

import argparse
import sys
import os
from music_genre_analysis import MusicGenreAnalyzer


def run_quick_analysis(n_clusters=10):
    """Run a quick analysis with reduced dataset."""
    print("Running QUICK analysis...")

    analyzer = MusicGenreAnalyzer()

    # Load data
    data = analyzer.load_and_preprocess_data()

    # Sample data for quick analysis
    if len(data) > 10000:
        analyzer.data = data.sample(n=5000, random_state=42)
        print(f"Sampled {len(analyzer.data)} tracks for quick analysis")

    # Basic EDA
    eda_results = analyzer.exploratory_data_analysis()

    # Prepare features and run clustering
    analyzer.prepare_features()
    analyzer.perform_clustering(n_clusters=n_clusters)

    # Evaluate results
    evaluation_results = analyzer.evaluate_clustering()

    # Create basic visualizations
    analyzer.create_visualizations()
    analyzer.create_clustering_visualizations()

    print("\nQuick analysis completed!")
    return analyzer


def run_full_analysis(n_clusters=10, run_experiments=False):
    """Run complete analysis with all features."""
    print("Running FULL analysis...")

    analyzer = MusicGenreAnalyzer()

    # Complete workflow
    data = analyzer.load_and_preprocess_data()
    eda_results = analyzer.exploratory_data_analysis()
    analyzer.create_visualizations()
    analyzer.prepare_features()
    analyzer.perform_clustering(n_clusters=n_clusters)
    evaluation_results = analyzer.evaluate_clustering()
    analyzer.create_clustering_visualizations()

    if run_experiments:
        print("\nRunning comprehensive experiments...")
        experiment_results = analyzer.run_experiments(
            train_test_splits=[0.5, 0.6, 0.7, 0.8],
            n_clusters_range=[5, 8, 10, 12, 15]
        )

    # Generate report and save results
    analyzer.generate_report()
    analyzer.save_results()

    print("\nFull analysis completed!")
    return analyzer


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Music Genre Discovery Analysis Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_analysis.py --quick
    python run_analysis.py --full --clusters 12
    python run_analysis.py --experiments
        """
    )

    parser.add_argument('--quick', action='store_true',
                       help='Run quick analysis with reduced dataset')
    parser.add_argument('--full', action='store_true',
                       help='Run complete analysis (default)')
    parser.add_argument('--clusters', type=int, default=10,
                       help='Number of clusters to use (default: 10)')
    parser.add_argument('--experiments', action='store_true',
                       help='Run comprehensive experiments')

    args = parser.parse_args()

    # Check if data directory exists
    data_dir = "Spotify/data/"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        print("Please ensure your Spotify dataset is in the correct location.")
        sys.exit(1)

    # Determine analysis type
    if args.quick:
        analyzer = run_quick_analysis(n_clusters=args.clusters)
    else:
        analyzer = run_full_analysis(n_clusters=args.clusters, run_experiments=args.experiments)

    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)

    if hasattr(analyzer, 'data'):
        print(f"Dataset size: {len(analyzer.data):,} tracks")
        print(f"Features analyzed: {len(analyzer.all_features)}")

    if hasattr(analyzer, 'evaluation_metrics') and not analyzer.evaluation_metrics.empty:
        print(f"Clustering algorithms tested: {len(analyzer.evaluation_metrics)}")
        best_algo = analyzer.evaluation_metrics.loc[
            analyzer.evaluation_metrics['Silhouette_Score'].idxmax(), 'Algorithm'
        ]
        best_score = analyzer.evaluation_metrics['Silhouette_Score'].max()
        print(f"Best performing algorithm: {best_algo} (Silhouette: {best_score:.4f})")

    print("\nOutput files generated:")
    print("- visualizations/: Data exploration plots")
    print("- clustering_results/: Clustering analysis plots")
    print("- results/: Numerical results and processed data")
    print("- analysis_report.html: Comprehensive HTML report")

    return analyzer


if __name__ == "__main__":
    main()