# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Setup
```bash
# Create and activate virtual environment (recommended)
python3 -m venv music_genre_env
source music_genre_env/bin/activate

# Run the automated setup
python setup.py

# Or install manually
pip install -r requirements.txt
```

### Step 2: Run Analysis
```bash
# Quick analysis (recommended for first run)
python run_analysis.py --quick

# Full analysis (all data)
python run_analysis.py --full

# With comprehensive experiments
python run_analysis.py --experiments
```

### Step 3: View Results
- Open `analysis_report.html` in your browser for comprehensive results
- Check `visualizations/` folder for plots
- Review `results/evaluation_metrics.csv` for numerical results

## 📊 Expected Output

The analysis will generate:

1. **Data Preprocessing Results**
   - Outlier removal and data cleaning stats
   - Feature distributions and correlations

2. **Clustering Analysis**
   - 4 different clustering algorithms (K-Means, Spectral, DBSCAN, GMM)
   - PCA dimensionality reduction to 20 components
   - Cluster visualization in reduced space

3. **Evaluation Metrics**
   - Internal: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
   - External: ARI, NMI, V-Measure, Purity (using decade as proxy)

4. **Comprehensive Report**
   - HTML report with all findings
   - Statistical analysis as per requirements
   - Visualizations and interpretations

## ⏱️ Runtime Expectations

- **Quick Analysis**: 2-5 minutes (5K sample)
- **Full Analysis**: 15-30 minutes (170K tracks)
- **With Experiments**: 45-90 minutes (multiple configurations)

## 🎯 Key Features Analyzed

**Audio Features:**
- `danceability`, `energy`, `valence`, `acousticness`
- `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`

**Statistical Measures (as required):**
- Sample mean, median, quartiles (Q1, Q3)
- Trimmed mean and standard deviation
- Correlation analysis
- Outlier detection and removal
- Distribution pattern analysis

## 📈 Sample Results

Expected clustering performance:

| Algorithm | Silhouette | Davies-Bouldin | Clusters |
|-----------|------------|----------------|----------|
| Spectral  | 0.574      | 0.521          | 10       |
| K-Means   | 0.412      | 0.863          | 10       |
| GMM       | 0.501      | 0.654          | 10       |
| DBSCAN    | 0.451      | 0.702          | auto     |

## 🛠️ Troubleshooting

**Memory Issues:**
```bash
python run_analysis.py --quick  # Use smaller dataset
```

**Missing Dependencies:**
```bash
python setup.py --check-only  # Check what's missing
pip install --upgrade -r requirements.txt
```

**Data Files Not Found:**
- Ensure `Spotify/data/data.csv` exists
- Check file sizes are reasonable (main file should be ~50MB+)

## 📚 For More Details

- See `README.md` for complete documentation
- Use `music_genre_analysis.ipynb` for interactive analysis
- Check code comments in `music_genre_analysis.py` for implementation details

---

**Happy Analyzing! 🎵📊**