Results:


GTZAN DATASET:

================================================================================
                    FINAL EXPERIMENT REPORT
================================================================================

1. DATASET INFORMATION
--------------------------------------------------------------------------------
   Dataset: features_30_sec.csv
   Total samples: 1000
   Number of features: 57
   Number of classes: 10
   Classes: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

2. EXPERIMENT CONFIGURATION
--------------------------------------------------------------------------------
   Train/Test splits tested: [0.5, 0.6, 0.7, 0.8]
   Random seeds used: [0, 42, 1337]
   Total experiments: 60
   Algorithms tested: KMeans, MiniBatchKMeans, Spectral, GMM, DBSCAN

3. DBSCAN PARAMETERS (AUTO-TUNED)
--------------------------------------------------------------------------------
   eps: 6.2636
   min_samples: 3

4. BEST OVERALL ALGORITHM (by Average Silhouette Score)
--------------------------------------------------------------------------------
   🏆 Winner: DBSCAN
   Metrics:
      silhouette: 0.2231
      davies_bouldin: 1.5211
      calinski_harabasz: 8.8049
      nmi: 0.0252
      ari: 0.0006
      acc_hungarian: 0.1137

5. BEST SINGLE CONFIGURATION (Highest Silhouette)
--------------------------------------------------------------------------------
   Algorithm: DBSCAN
   Train Ratio: 0.5
   Random Seed: 42
   Metrics:
      silhouette: 0.3460
      davies_bouldin: 1.6917
      calinski_harabasz: 15.7786
      nmi: 0.0201
      ari: 0.0004
      acc_hungarian: 0.1120

6. ALGORITHM RANKINGS (by Average Metrics)
--------------------------------------------------------------------------------
   1. KMeans (Composite Score: 5.9303)
   2. Spectral (Composite Score: 5.5184)
   3. MiniBatchKMeans (Composite Score: 5.4905)
   4. GMM (Composite Score: 4.1232)
   5. DBSCAN (Composite Score: 1.5940)

✅ Comprehensive report saved to: /home/anirudh-sharma/Desktop/GITHUB Music Genere/Results/GTZAN/experiment_report.txt

================================================================================
ALL RESULTS SAVED TO: /home/anirudh-sharma/Desktop/GITHUB Music Genere/Results/GTZAN
================================================================================
================================================================================
📁 ALL GENERATED FILES
================================================================================

Output Directory: /home/anirudh-sharma/Desktop/GITHUB Music Genere/Results/GTZAN

📊 CSV FILES:
--------------------------------------------------------------------------------
   best_configurations.csv                            (    1.12 KB)
   clustering_results_detailed.csv                    (   10.12 KB)
   clustering_results_summary.csv                     (    4.82 KB)
   dbscan_detailed_results.csv                        (    2.58 KB)

🖼️  VISUALIZATION FILES:
--------------------------------------------------------------------------------
   algorithm_performance_ranking.png                  (  192.19 KB)
   algorithm_stability.png                            (  195.82 KB)
   boxplot_all_metrics.png                            (  167.52 KB)
   dbscan_analysis.png                                (  351.72 KB)
   distribution_by_algorithm.png                      (  361.24 KB)
   heatmaps_all_metrics.png                           (  755.62 KB)
   metrics_comparison.png                             (  349.28 KB)
   radar_chart_comparison.png                         (  723.37 KB)

📝 REPORT FILES:
--------------------------------------------------------------------------------
   experiment_report.txt                              (    1.38 KB)

================================================================================
✅ TOTAL FILES GENERATED: 13
   CSV: 4, PNG: 8, TXT: 1
================================================================================

📋 PREVIEW: First 5 rows of detailed results
--------------------------------------------------------------------------------
algo	train_ratio	random_state	silhouette	nmi	ari	acc_hungarian	n_clusters_found
0	KMeans	0.5	0	NaN	0.326183	0.176756	0.346	10
1	MiniBatchKMeans	0.5	0	0.145843	0.339340	0.166162	0.348	10
2	Spectral	0.5	0	0.079274	0.344100	0.165571	0.338	10
3	GMM	0.5	0	-0.015542	0.228128	0.104352	0.304	6
4	DBSCAN	0.5	0	0.195857	0.034238	0.001164	0.118	2


🎓 Conclusions
Key Findings:
Unsupervised Clustering: Successfully performed clustering on MSD dataset without genre labels
Algorithm Comparison: Evaluated 5 different clustering algorithms across multiple metrics
Robustness Testing: Used multiple train/test splits and random seeds for reliable results
Auto-tuning: Automatically determined optimal number of clusters and DBSCAN parameters
Metrics Explained:
Silhouette Score (0 to 1, higher is better): Measures how similar objects are to their own cluster vs other clusters
Davies-Bouldin Index (≥0, lower is better): Average similarity between each cluster and its most similar cluster
Calinski-Harabasz Score (higher is better): Ratio of between-cluster to within-cluster dispersion
Next Steps:
Explore discovered clusters - what makes them different?
Try different feature engineering approaches
Experiment with dimensionality reduction before clustering
Consider ensemble clustering methods
All results are logged to W&B for interactive exploration! 🚀


