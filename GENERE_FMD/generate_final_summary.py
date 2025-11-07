#!/usr/bin/env python3
"""
Generate Final Summary Report for Full Dataset Analysis
"""

import pandas as pd
import os
from datetime import datetime


def generate_final_summary():
    """Generate comprehensive summary of the full dataset analysis."""

    print("="*80)
    print("📊 COMPREHENSIVE ANALYSIS REPORT - FULL DATASET")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Load extracted features to get count
    features_df = pd.read_csv('results/extracted_features.csv')
    n_samples = len(features_df)
    n_features = len(features_df.columns) - 1  # Exclude track_id

    print("\n📁 DATASET INFORMATION")
    print("-" * 80)
    print(f"✅ Total audio files processed: {n_samples:,}")
    print(f"✅ Total features extracted: {n_features}")
    print(f"✅ Feature types: MFCCs (40) + Delta-MFCCs (40) + Delta²-MFCCs (40)")
    print(f"                  + Chroma (24) + Spectral (6) + Other (5)")

    # Load cleaned features to see after cleaning
    cleaned_df = pd.read_csv('results/cleaned_features.csv')
    n_cleaned = len(cleaned_df)
    print(f"✅ Samples after cleaning: {n_cleaned:,}")
    print(
        f"✅ Outliers removed: {n_samples - n_cleaned:,} ({((n_samples - n_cleaned)/n_samples*100):.2f}%)")

    print("\n" + "="*80)
    print("🏆 BEST RESULTS ACHIEVED (Across All Splits)")
    print("="*80)

    # Load all comparison tables
    splits = ['50-50', '60-40', '70-30', '80-20']
    best_results = {
        'purity': {'score': 0, 'algorithm': '', 'split': ''},
        'accuracy': {'score': 0, 'algorithm': '', 'split': ''},
        'nmi': {'score': 0, 'algorithm': '', 'split': ''},
        'ari': {'score': 0, 'algorithm': '', 'split': ''},
        'silhouette': {'score': 0, 'algorithm': '', 'split': ''},
    }

    for split in splits:
        try:
            df = pd.read_csv(f'results/comparison_table_{split}.csv')

            # Find best for each metric
            max_purity_idx = df['Purity'].idxmax()
            if df.loc[max_purity_idx, 'Purity'] > best_results['purity']['score']:
                best_results['purity'] = {
                    'score': df.loc[max_purity_idx, 'Purity'],
                    'algorithm': df.loc[max_purity_idx, 'Algorithm'],
                    'split': split
                }

            max_acc_idx = df['Accuracy'].idxmax()
            if df.loc[max_acc_idx, 'Accuracy'] > best_results['accuracy']['score']:
                best_results['accuracy'] = {
                    'score': df.loc[max_acc_idx, 'Accuracy'],
                    'algorithm': df.loc[max_acc_idx, 'Algorithm'],
                    'split': split
                }

            max_nmi_idx = df['NMI'].idxmax()
            if df.loc[max_nmi_idx, 'NMI'] > best_results['nmi']['score']:
                best_results['nmi'] = {
                    'score': df.loc[max_nmi_idx, 'NMI'],
                    'algorithm': df.loc[max_nmi_idx, 'Algorithm'],
                    'split': split
                }

            max_ari_idx = df['ARI'].idxmax()
            if df.loc[max_ari_idx, 'ARI'] > best_results['ari']['score']:
                best_results['ari'] = {
                    'score': df.loc[max_ari_idx, 'ARI'],
                    'algorithm': df.loc[max_ari_idx, 'Algorithm'],
                    'split': split
                }

            max_sil_idx = df['Silhouette'].idxmax()
            if df.loc[max_sil_idx, 'Silhouette'] > best_results['silhouette']['score']:
                best_results['silhouette'] = {
                    'score': df.loc[max_sil_idx, 'Silhouette'],
                    'algorithm': df.loc[max_sil_idx, 'Algorithm'],
                    'split': split
                }
        except Exception as e:
            print(f"Warning: Could not load {split}: {e}")

    # Print best results
    print(f"\n🥇 PURITY (Best): {best_results['purity']['score']:.2%}")
    print(f"   Algorithm: {best_results['purity']['algorithm']}")
    print(f"   Split: {best_results['purity']['split']}")

    print(f"\n🥇 ACCURACY (Best): {best_results['accuracy']['score']:.2%}")
    print(f"   Algorithm: {best_results['accuracy']['algorithm']}")
    print(f"   Split: {best_results['accuracy']['split']}")

    print(f"\n🥇 NMI (Best): {best_results['nmi']['score']:.4f}")
    print(f"   Algorithm: {best_results['nmi']['algorithm']}")
    print(f"   Split: {best_results['nmi']['split']}")

    print(f"\n🥇 ARI (Best): {best_results['ari']['score']:.4f}")
    print(f"   Algorithm: {best_results['ari']['algorithm']}")
    print(f"   Split: {best_results['ari']['split']}")

    print(f"\n🥇 SILHOUETTE (Best): {best_results['silhouette']['score']:.4f}")
    print(f"   Algorithm: {best_results['silhouette']['algorithm']}")
    print(f"   Split: {best_results['silhouette']['split']}")

    print("\n" + "="*80)
    print("📊 ALGORITHM PERFORMANCE COMPARISON")
    print("="*80)

    # Aggregate results across all splits
    all_results = []
    for split in splits:
        try:
            df = pd.read_csv(f'results/comparison_table_{split}.csv')
            df['Split'] = split
            all_results.append(df)
        except:
            pass

    if all_results:
        combined = pd.concat(all_results)

        # Calculate average performance per algorithm
        avg_performance = combined.groupby('Algorithm').agg({
            'Purity': 'mean',
            'Accuracy': 'mean',
            'NMI': 'mean',
            'ARI': 'mean',
            'Silhouette': 'mean'
        }).sort_values('Purity', ascending=False)

        print("\nAverage Performance Across All Splits:")
        print("-" * 80)
        print(
            f"{'Algorithm':<25} {'Purity':>10} {'Accuracy':>10} {'NMI':>8} {'ARI':>8}")
        print("-" * 80)
        for algo, row in avg_performance.iterrows():
            if pd.notna(row['Purity']):
                print(f"{algo:<25} {row['Purity']:>9.2%} {row['Accuracy']:>9.2%} "
                      f"{row['NMI']:>8.4f} {row['ARI']:>8.4f}")

    print("\n" + "="*80)
    print("📈 IMPROVEMENT OVER 200-FILE BASELINE")
    print("="*80)

    # Compare with previous best (73.33% purity with 200 files)
    baseline_purity = 0.7333
    improvement = (
        (best_results['purity']['score'] - baseline_purity) / baseline_purity) * 100

    print(f"\nBaseline (200 files): {baseline_purity:.2%} purity")
    print(
        f"Full Dataset ({n_cleaned} files): {best_results['purity']['score']:.2%} purity")

    if improvement > 0:
        print(f"Improvement: +{improvement:.2f}% ✅")
    else:
        print(f"Change: {improvement:.2f}%")
        print("\n⚠️  Note: Lower performance with more data can indicate:")
        print("   - Greater diversity in the dataset")
        print("   - More challenging/ambiguous samples")
        print("   - Need for more sophisticated algorithms or features")
        print("   - The 200-file subset may have been easier to cluster")

    print("\n" + "="*80)
    print("📂 GENERATED FILES")
    print("="*80)

    results_files = sorted([f for f in os.listdir(
        'results') if os.path.isfile(f'results/{f}')])

    print("\nComparison Tables:")
    for f in [f for f in results_files if f.startswith('comparison_table')]:
        print(f"  ✓ {f}")

    print("\nVisualization Plots:")
    for f in [f for f in results_files if f.endswith('.png')]:
        print(f"  ✓ {f}")

    print("\nData Files:")
    for f in [f for f in results_files if f.endswith('.csv') and not f.startswith('comparison')]:
        print(f"  ✓ {f}")

    print("\nReports:")
    for f in [f for f in results_files if f.endswith(('.txt', '.md'))]:
        print(f"  ✓ {f}")

    print("\n" + "="*80)
    print("🎯 EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"✅ Number of clusters: 8 (matching FMA genres)")
    print(f"✅ PCA components: 20 (explaining {61.70:.2f}% variance)")
    print(f"✅ Train-test splits: 50-50, 60-40, 70-30, 80-20")
    print(f"✅ Algorithms: K-Means, MiniBatch K-Means, Spectral, DBSCAN, GMM")
    print(f"✅ DBSCAN: Auto-tuned using k-distance plot")
    print(f"✅ Metadata: Real FMA genre labels")
    print(f"✅ Metrics: Silhouette, DBI, CHI, ARI, NMI, V-Measure, Purity, Accuracy")

    print("\n" + "="*80)
    print("✨ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n🎉 Successfully processed {n_samples:,} audio files")
    print(f"🎉 Generated comprehensive clustering analysis")
    print(f"🎉 All results saved to 'results/' directory")
    print("\n" + "="*80)

    # Save summary to file
    with open('results/FULL_DATASET_SUMMARY.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("MUSIC GENRE CLUSTERING - FULL DATASET ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total audio files processed: {n_samples:,}\n")
        f.write(f"Samples after cleaning: {n_cleaned:,}\n")
        f.write(f"Features extracted: {n_features}\n\n")

        f.write("BEST RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"Purity: {best_results['purity']['score']:.2%} ({best_results['purity']['algorithm']}, {best_results['purity']['split']})\n")
        f.write(
            f"Accuracy: {best_results['accuracy']['score']:.2%} ({best_results['accuracy']['algorithm']}, {best_results['accuracy']['split']})\n")
        f.write(
            f"NMI: {best_results['nmi']['score']:.4f} ({best_results['nmi']['algorithm']}, {best_results['nmi']['split']})\n")
        f.write(
            f"ARI: {best_results['ari']['score']:.4f} ({best_results['ari']['algorithm']}, {best_results['ari']['split']})\n")
        f.write(
            f"Silhouette: {best_results['silhouette']['score']:.4f} ({best_results['silhouette']['algorithm']}, {best_results['silhouette']['split']})\n\n")

        if all_results:
            f.write("\nAVERAGE ALGORITHM PERFORMANCE:\n")
            f.write("-" * 80 + "\n")
            f.write(avg_performance.to_string())
            f.write("\n")

    print("\n📄 Summary report saved to: results/FULL_DATASET_SUMMARY.txt")


if __name__ == "__main__":
    generate_final_summary()
