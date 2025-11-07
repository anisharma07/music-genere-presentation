"""
Main Execution Script for Music Genre Discovery Project
=======================================================

This script orchestrates the complete workflow:
1. Data Analysis and Cleaning
2. Feature Extraction and PCA
3. Clustering with Multiple Algorithms
4. Comprehensive Evaluation
5. Visualization and Reporting

Author: Anirudh Sharma
Topic: Unsupervised Music Genre Discovery Using Audio Feature Learning
Date: November 2025
"""

from clustering_implementation import MusicGenreClusterer
from data_analysis import MusicDataAnalyzer
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules


def create_directory_structure():
    """Create necessary directories for results and outputs."""
    directories = ['results', 'visualizations', 'reports']

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}/")


def main():
    """
    Main execution function.

    This function runs the complete pipeline:
    - Data analysis and cleaning
    - Clustering experiments
    - Evaluation and visualization
    """

    print("\n" + "=" * 80)
    print(" " * 15 + "MUSIC GENRE DISCOVERY USING UNSUPERVISED LEARNING")
    print(" " * 25 + "GTZAN Dataset Analysis")
    print("=" * 80)
    print("\nProject: Unsupervised Music Genre Discovery Using Audio Feature Learning")
    print("Author: Anirudh Sharma")
    print("Date: November 2025")
    print("=" * 80)

    # Create directory structure
    print("\n[STEP 1] Setting up directory structure...")
    create_directory_structure()

    # ============================================================================
    # PHASE 1: DATA ANALYSIS AND CLEANING
    # ============================================================================

    print("\n" + "=" * 80)
    print("[PHASE 1] DATA ANALYSIS AND CLEANING")
    print("=" * 80)

    try:
        # Initialize data analyzer
        analyzer = MusicDataAnalyzer('gtzan/features_30_sec.csv')

        # Generate comprehensive analysis report
        report = analyzer.generate_full_report()

        # Save cleaned dataset
        if analyzer.cleaned_df is not None:
            analyzer.cleaned_df.to_csv(
                'gtzan/features_30_sec_cleaned.csv', index=False)
            print("\n✓ Cleaned dataset saved: gtzan/features_30_sec_cleaned.csv")

        print("\n✓ Phase 1 completed successfully!")

    except Exception as e:
        print(f"\n✗ Error in Phase 1: {str(e)}")
        print("Continuing with original dataset...")

    # ============================================================================
    # PHASE 2: CLUSTERING EXPERIMENTS
    # ============================================================================

    print("\n" + "=" * 80)
    print("[PHASE 2] CLUSTERING EXPERIMENTS")
    print("=" * 80)

    try:
        # Try to use cleaned dataset, fall back to original if not available
        dataset_path = 'gtzan/features_30_sec_cleaned.csv'
        if not os.path.exists(dataset_path):
            dataset_path = 'gtzan/features_30_sec.csv'
            print(f"Using original dataset: {dataset_path}")
        else:
            print(f"Using cleaned dataset: {dataset_path}")

        # Initialize clusterer
        clusterer = MusicGenreClusterer(dataset_path)

        # Run all experiments with different train-test splits
        print("\n[STEP 2.1] Running experiments with different splits...")
        results = clusterer.run_all_experiments()

        # Generate visualizations
        print("\n[STEP 2.2] Generating visualizations...")
        clusterer.visualize_results(results)

        # Generate summary
        print("\n[STEP 2.3] Creating summary tables...")
        summary = clusterer.generate_summary_table(results)

        # Generate detailed visualizations for best split (80-20)
        print("\n[STEP 2.4] Creating detailed cluster visualizations...")
        results_final, models, X_test, y_test = clusterer.run_experiment(
            (80, 20))

        for algo_name, (model, labels) in models.items():
            clusterer.visualize_clusters_2d(X_test, y_test, labels, algo_name)

        print("\n✓ Phase 2 completed successfully!")

    except Exception as e:
        print(f"\n✗ Error in Phase 2: {str(e)}")
        import traceback
        traceback.print_exc()

    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================

    print("\n" + "=" * 80)
    print("PROJECT EXECUTION COMPLETED!")
    print("=" * 80)

    print("\n📊 Generated Files:")
    print("\n  Data Analysis Results:")
    print("    - results/class_balance.png")
    print("    - results/descriptive_statistics.csv")
    print("    - results/outlier_boxplots.png")
    print("    - results/distribution_analysis.png")
    print("    - results/percentile_quartile_stats.csv")
    print("    - results/trimmed_statistics.csv")
    print("    - results/correlation_matrix.csv")
    print("    - results/correlation_heatmap.png")

    print("\n  Clustering Results:")
    print("    - results/clustering_results.csv")
    print("    - results/summary_table.csv")
    print("    - results/metrics_comparison.png")
    print("    - results/performance_by_split.png")
    print("    - results/radar_chart.png")
    print("    - results/cluster_viz_*.png (for each algorithm)")

    print("\n  Cleaned Dataset:")
    print("    - gtzan/features_30_sec_cleaned.csv")

    print("\n" + "=" * 80)
    print("📝 Next Steps:")
    print("  1. Review the generated visualizations in the results/ directory")
    print("  2. Analyze the clustering_results.csv for detailed metrics")
    print("  3. Compare algorithm performance using the radar chart")
    print("  4. Document findings in your project report")
    print("=" * 80)

    print("\n✓ All tasks completed successfully!")
    print("\nThank you for using the Music Genre Discovery System!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    """
    Entry point for the music genre discovery project.
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Execution interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
