# Quick Reference Guide

## Installation Commands

```bash
# 1. Navigate to project
cd "/home/anirudh-sharma/Desktop/Music Genere/GENERE_MSD"

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Test setup
python test_setup.py
```

## Running the Pipeline

### Option 1: Quick Start (Recommended)
```bash
./run.sh
```

### Option 2: Manual Run
```bash
source venv/bin/activate
python main.py
```

### Option 3: Step-by-Step
```bash
source venv/bin/activate
python feature_extractor.py    # Step 1: Extract features
python data_cleaner.py          # Step 2: Clean data
python clustering.py            # Step 3: Cluster
python evaluation.py            # Step 4: Evaluate
python visualization.py         # Step 5: Visualize
```

## File Locations

### Code Files
- `config.py` - Configuration settings
- `feature_extractor.py` - Feature extraction
- `data_cleaner.py` - Data cleaning
- `clustering.py` - Clustering algorithms
- `evaluation.py` - Metrics
- `visualization.py` - Plots
- `main.py` - Main pipeline

### Output Files
- `output/extracted_features.csv` - Raw features
- `output/cleaned_features.csv` - Cleaned features
- `output/clustered_data.csv` - With cluster labels
- `output/results/evaluation_metrics.csv` - Metrics table
- `output/results/final_report.txt` - Summary
- `output/plots/*.png` - All visualizations
- `output/models/*.pkl` - Trained models

## Configuration Options

### Limit Files (for testing)
Edit `main.py`:
```python
MAX_FILES = 100  # Process only 100 files
```

### Change Number of Clusters
Edit `main.py`:
```python
N_CLUSTERS_LIST = [5, 10, 15]
```

### Enable Cross-Validation
Edit `main.py`:
```python
RUN_CROSS_VALIDATION = True
```

### Change PCA Components
Edit `config.py`:
```python
PCA_COMPONENTS = 15
```

### Adjust Data Cleaning
Edit `config.py`:
```python
DATA_CLEANING = {
    'remove_outliers': True,
    'outlier_method': 'zscore',  # or 'iqr'
    'handle_missing': 'median'    # or 'mean' or 'drop'
}
```

## Algorithms Included

1. **K-Means** - Fast, works with spherical clusters
2. **MiniBatch K-Means** - Faster variant
3. **Spectral Clustering** - Non-convex clusters
4. **DBSCAN** - Density-based, automatic cluster count
5. **GMM** - Gaussian Mixture Model, soft assignments

## Evaluation Metrics

### Internal Metrics (no ground truth needed)
- **Silhouette Score**: Higher is better (0.5+ is good)
- **Davies-Bouldin Index**: Lower is better (<1.0 is good)
- **Calinski-Harabasz Index**: Higher is better (>300 is good)

### External Metrics (if ground truth available)
- **Adjusted Rand Index (ARI)**: Higher is better
- **Normalized Mutual Information (NMI)**: Higher is better
- **V-Measure**: Higher is better
- **Purity Index**: Higher is better

## Features Extracted (113 total)

### Basic Features (13)
- duration, tempo, loudness, key, mode, time_signature
- energy, loudness_max_mean, loudness_max_std
- segment_density

### Timbre Features (48)
- 12 timbre coefficients × 4 statistics (mean, std, min, max)

### Pitch Features (48)
- 12 pitch coefficients × 4 statistics (mean, std, min, max)

### Metadata (3)
- track_id, artist_name, title

## Common Commands

### Check Python Version
```bash
python3 --version
```

### Verify Packages
```bash
python3 -c "import h5py, numpy, pandas, sklearn; print('OK')"
```

### Count HDF5 Files
```bash
find "million song" -name "*.h5" | wc -l
```

### View First HDF5 File
```bash
find "million song" -name "*.h5" | head -1
```

### Check Output Size
```bash
du -sh output/
```

### View Logs
```bash
tail -f output/pipeline.log
```

### Clean Outputs
```bash
rm -rf output/*
```

## Troubleshooting

### "Module not found"
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "No HDF5 files found"
- Check `DATA_DIR` in `config.py`
- Verify dataset is extracted

### "Out of memory"
- Reduce `MAX_FILES` in `main.py`
- Reduce `PCA_COMPONENTS` in `config.py`

### "Pipeline too slow"
- Set `MAX_FILES = 1000` for testing
- Skip spectral clustering (comment out in `clustering.py`)

## Expected Runtime

| Dataset Size | Time | Memory |
|--------------|------|--------|
| 100 files | ~2 min | 500MB |
| 1,000 files | ~10 min | 1GB |
| 10,000 files | ~30 min | 2GB |
| Full dataset | ~2 hours | 4GB |

## Results Interpretation

### Good Clustering Results
- Silhouette Score > 0.4
- Davies-Bouldin < 1.5
- Calinski-Harabasz > 200
- Clear cluster separation in t-SNE plot

### Poor Clustering Results
- Silhouette Score < 0.2
- Davies-Bouldin > 2.0
- Calinski-Harabasz < 100
- Overlapping clusters in t-SNE plot

## Next Steps After Running

1. Check `output/results/final_report.txt`
2. View `output/plots/metrics_comparison.png`
3. Analyze `output/plots/tsne_visualization.png`
4. Review `output/results/evaluation_metrics.csv`
5. Include plots in your report
6. Copy metrics to your tables

## Complete Workflow Summary

```
1. Extract Features → extracted_features.csv
2. Clean Data → cleaned_features.csv + statistics + plots
3. Apply Clustering → clustered_data.csv + models
4. Evaluate → evaluation_metrics.csv
5. Visualize → plots/*.png
6. Report → final_report.txt
```

## Getting Help

1. Check `output/pipeline.log` for errors
2. Run `python test_setup.py` to verify setup
3. Review `README.md` for detailed documentation
4. Check error messages in terminal

## Export for Report

### Tables for Report
- Copy from `evaluation_metrics.csv`
- Copy from `descriptive_statistics.csv`

### Figures for Report
- All PNG files in `output/plots/`
- Recommended: metrics_comparison, tsne_visualization, correlation_heatmap

### Text for Report
- Use `final_report.txt` as template
- Include statistics from `descriptive_statistics.csv`
