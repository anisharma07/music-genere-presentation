"""
Continue Analysis Script
Continues the pipeline from existing extracted features, skipping feature extraction.
"""

from evaluation import ClusteringEvaluator
from clustering import ClusteringEngine
from data_analysis import DataAnalyzer
from main import MusicGenreClusteringPipeline
import os
import warnings
warnings.filterwarnings('ignore')


def continue_from_features():
    """Continue pipeline from existing extracted features."""

    # Configuration
    OUTPUT_DIR = 'results'
    FEATURES_PATH = 'results/extracted_features.csv'
    N_CLUSTERS = 8  # Match FMA genre count

    # Check if extracted features exist
    if not os.path.exists(FEATURES_PATH):
        print(f"❌ Error: {FEATURES_PATH} not found!")
        print("Please run feature extraction first.")
        return

    print("="*80)
    print("CONTINUING MUSIC GENRE CLUSTERING PIPELINE")
    print("="*80)
    print(f"Using existing features from: {FEATURES_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)

    # Create pipeline instance
    pipeline = MusicGenreClusteringPipeline(
        data_path='fma_small',
        output_dir=OUTPUT_DIR
    )

    # Step 2: Data Analysis and Cleaning
    print("\n" + "="*80)
    print("STEP 2: DATA ANALYSIS AND CLEANING")
    print("="*80)
    cleaned_path = pipeline.step2_data_analysis(FEATURES_PATH)

    # Step 3: Clustering Experiments
    print("\n" + "="*80)
    print("STEP 3: CLUSTERING EXPERIMENTS")
    print("="*80)
    engine, all_results = pipeline.step3_clustering_experiments(
        cleaned_path, n_clusters=N_CLUSTERS
    )

    # Step 4: Evaluation
    print("\n" + "="*80)
    print("STEP 4: EVALUATION")
    print("="*80)
    all_evaluations = pipeline.step4_evaluation(
        engine, all_results, use_synthetic_labels=False
    )

    # Step 5: Generate Comparison Tables
    print("\n" + "="*80)
    print("STEP 5: GENERATE COMPARISON TABLES")
    print("="*80)
    pipeline.step5_generate_comparison_tables(all_evaluations)

    # Step 6: Visualize Results
    print("\n" + "="*80)
    print("STEP 6: VISUALIZE RESULTS")
    print("="*80)
    pipeline.step6_visualize_results(all_evaluations)

    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for file in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {file}")

    # Print summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)

    # Count processed files
    import pandas as pd
    df = pd.read_csv(FEATURES_PATH)
    print(f"✅ Total audio files processed: {len(df)}")
    # Exclude track_id
    print(f"✅ Total features extracted: {len(df.columns) - 1}")
    print(f"✅ Number of clusters: {N_CLUSTERS}")
    print(f"✅ PCA components: 20")
    print(f"✅ Train-test splits tested: 50-50, 60-40, 70-30, 80-20")
    print(f"✅ Algorithms tested: 5 (K-Means, MiniBatch K-Means, Spectral, DBSCAN, GMM)")
    print(f"✅ Metrics computed: Silhouette, DBI, CHI, ARI, NMI, V-Measure, Purity, Accuracy")
    print(f"✅ Metadata: REAL FMA genre labels")
    print("="*80)

    return all_evaluations


if __name__ == "__main__":
    continue_from_features()
