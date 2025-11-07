# Quick Start Guide
# Music Genre Discovery Using Unsupervised Learning

## Installation

### Step 1: Install Required Packages
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

Or use the provided setup script:
```bash
chmod +x setup.sh
./setup.sh
```

Or use requirements file:
```bash
pip install -r requirements.txt
```

## Running the Project

### Option 1: Complete Pipeline (Recommended)
```bash
python main.py
```

This will:
1. Analyze and clean the data
2. Run clustering experiments with multiple splits
3. Generate all visualizations
4. Create summary reports

Expected runtime: 5-15 minutes depending on your system

### Option 2: Individual Modules

**Data Analysis Only:**
```bash
python data_analysis.py
```

**Clustering Experiments:**
```bash
python clustering_implementation.py
```

**Cross-Validation:**
```bash
python cross_validation.py
```

## Expected Outputs

After running `main.py`, you should see:

### In `results/` directory:
- `class_balance.png` - Genre distribution
- `descriptive_statistics.csv` - Statistical summary
- `outlier_boxplots.png` - Outlier visualizations
- `distribution_analysis.png` - Feature distributions
- `correlation_matrix.csv` - Feature correlations
- `correlation_heatmap.png` - Correlation visualization
- `clustering_results.csv` - All experiment results
- `summary_table.csv` - Average performance
- `metrics_comparison.png` - Algorithm comparisons
- `performance_by_split.png` - Performance trends
- `radar_chart.png` - Multi-metric comparison
- `cluster_viz_*.png` - 2D cluster visualizations

### In `gtzan/` directory:
- `features_30_sec_cleaned.csv` - Cleaned dataset

## Troubleshooting

### Import Errors
If you see import errors:
```bash
pip install --upgrade numpy pandas scikit-learn matplotlib seaborn scipy
```

### Memory Issues
If you encounter memory errors:
1. Edit `config.py` and reduce `N_PCA_COMPONENTS` to 10
2. Use MiniBatch K-Means instead of regular K-Means
3. Process one split at a time

### DBSCAN Issues
If DBSCAN finds too many noise points:
1. Edit `config.py`
2. Decrease `DBSCAN_EPS` to 2.0
3. Decrease `DBSCAN_MIN_SAMPLES` to 3

## Customization

### Change Number of Clusters
Edit `config.py`:
```python
N_CLUSTERS = 10  # Change to desired number
```

### Modify PCA Components
Edit `config.py`:
```python
N_PCA_COMPONENTS = 20  # Adjust as needed
```

### Add More Train-Test Splits
Edit `config.py`:
```python
SPLIT_RATIOS = [
    (50, 50),
    (60, 40),
    (70, 30),
    (80, 20),
    (90, 10)  # Add new split
]
```

## Understanding the Results

### Internal Metrics (Unsupervised)
- **Silhouette Score**: [-1, 1] - Higher is better
  - > 0.5: Strong clusters
  - 0.25-0.5: Weak clusters
  - < 0.25: No substantial structure

- **Davies-Bouldin Index**: [0, ∞] - LOWER is better
  - < 1.0: Good separation
  - > 2.0: Poor separation

- **Calinski-Harabasz**: [0, ∞] - Higher is better
  - Higher values indicate better-defined clusters

### External Metrics (Supervised)
- **NMI, ARI, V-Measure**: [0, 1] - Higher is better
  - > 0.7: Excellent agreement with ground truth
  - 0.5-0.7: Good agreement
  - < 0.5: Weak agreement

- **Cluster Accuracy**: [0, 1] - Higher is better
  - Percentage of correctly assigned samples

## Project Structure
```
.
├── main.py                          # Main execution script
├── data_analysis.py                 # Data analysis module
├── clustering_implementation.py     # Clustering algorithms
├── cross_validation.py              # Cross-validation
├── utils.py                         # Utility functions
├── config.py                        # Configuration
├── requirements.txt                 # Dependencies
├── README.md                        # Full documentation
├── QUICKSTART.md                    # This file
├── gtzan/                          # Dataset
│   └── features_30_sec.csv
└── results/                        # Generated outputs
```

## Getting Help

1. Check the full README.md for detailed documentation
2. Review the code comments in each module
3. Examine the generated visualizations
4. Check the TO_DO.md for project requirements

## Next Steps

After running the code:
1. Review generated visualizations in `results/`
2. Analyze `clustering_results.csv` for detailed metrics
3. Compare algorithms using the radar chart
4. Document findings in your project report
5. Use the LaTeX table for academic writing

## Example Commands

### View Configuration
```bash
python config.py
```

### Test Utilities
```bash
python utils.py
```

### Check Data
```bash
python -c "import pandas as pd; df = pd.read_csv('gtzan/features_30_sec.csv'); print(df.head())"
```

## Performance Tips

1. **Speed up execution**: Use MiniBatch K-Means
2. **Reduce memory usage**: Lower PCA components
3. **Faster DBSCAN**: Increase `eps` parameter
4. **Parallel processing**: Set `N_JOBS = -1` in config

## Common Issues and Solutions

### Issue: "No module named 'sklearn'"
**Solution:**
```bash
pip install scikit-learn
```

### Issue: Plots not showing
**Solution:** Plots are automatically saved to `results/` directory

### Issue: "FileNotFoundError: gtzan/features_30_sec.csv"
**Solution:** Ensure you're in the correct directory with the dataset

### Issue: DBSCAN finds 0 clusters
**Solution:** Adjust DBSCAN parameters in `config.py`:
```python
DBSCAN_EPS = 2.0  # Decrease
DBSCAN_MIN_SAMPLES = 3  # Decrease
```

---

**For full documentation, see README.md**

Happy Clustering! 🎵🎶
