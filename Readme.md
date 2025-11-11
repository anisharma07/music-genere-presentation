# 🎵 Unsupervised Music Genre Discovery Using Audio Feature Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Overview](#-overview)
- [Datasets](#-datasets)
- [Project Structure](#-project-structure)
- [Features & Methodology](#-features--methodology)
- [Clustering Algorithms](#-clustering-algorithms)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Evaluation Metrics](#-evaluation-metrics)
- [Key Findings](#-key-findings)
- [Future Work](#-future-work)

---

## 📖 Overview

This project explores **unsupervised machine learning techniques** for discovering hidden patterns and clusters in music genres based on audio features. By applying multiple clustering algorithms across four diverse music datasets, we analyze the effectiveness of different approaches in genre classification without labeled training data.

### 🎯 Objectives
- Apply unsupervised learning algorithms to discover hidden genre clusters
- Perform comprehensive dataset analysis, feature extraction, and model evaluation
- Compare clustering algorithms across different datasets using multiple evaluation metrics
- Identify optimal configurations for music genre clustering

---

## 🗂️ Datasets

The project utilizes four diverse music datasets:

| Dataset | Tracks | Genres | Features | Source |
|---------|--------|--------|----------|--------|
| **GTZAN** | 1,000 | 10 | 57 | Audio clips (30s) |
| **FMA-Small** | 8,000 | 8 | Multiple | Free Music Archive |
| **Million Song Dataset (MSD)** | Variable | Regional | Audio features | Echo Nest API |
| **Spotify Tracks** | Variable | Multiple | Audio features | Spotify API |

### Genre Classes (GTZAN)
`blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`

---

## 📁 Project Structure

```
.
├── 1.ml-gtzan-genere.ipynb      # GTZAN dataset analysis
├── 2.FMA_Music_Genre.ipynb      # FMA dataset analysis
├── 3.msd-notebook.ipynb         # Million Song Dataset analysis
├── 4-spotify-notebook.ipynb     # Spotify dataset analysis
├── data/                        # Raw and processed data
│   ├── fma.csv
│   ├── gtzan.csv
│   ├── msd.csv
│   └── spotify.csv
├── results/                     # Experiment results and visualizations
│   ├── fma/
│   ├── gtzan/
│   ├── msd/
│   └── spotify/
├── models/                      # Saved model configurations
├── Presentation/                # LaTeX presentation files
├── IEEE_Report.pdf              # Detailed project report
└── README.md                    # This file
```

---

## 🔬 Features & Methodology

### Audio Feature Extraction (using Librosa)
- **MFCC** (Mel-Frequency Cepstral Coefficients): 20-40 coefficients
- **Chroma Features**: Pitch class representation
- **Spectral Features**:
  - Spectral Centroid
  - Spectral Bandwidth
  - Spectral Roll-off
- **Zero Crossing Rate**: Signal frequency estimation
- **Tempo & Rhythm**: Beat tracking and rhythmic patterns

### Data Preprocessing
- ✅ Descriptive statistics (mean, median, quartiles, IQR)
- ✅ Outlier detection and removal using boxplots
- ✅ Missing value imputation (mean replacement)
- ✅ Correlation analysis
- ✅ Distribution and skewness analysis
- ✅ Feature normalization and scaling

### Dimensionality Reduction
- **PCA** (Principal Component Analysis): Retain 95% variance
- **t-SNE** / **UMAP**: For 2D/3D visualization

---

## 🤖 Clustering Algorithms

Five clustering algorithms were evaluated:

| Algorithm | Type | Key Parameters |
|-----------|------|----------------|
| **K-Means** | Centroid-based | n_clusters=10 |
| **MiniBatch K-Means** | Centroid-based | n_clusters=10, batch_size |
| **Spectral Clustering** | Graph-based | n_clusters=10, affinity |
| **GMM** (Gaussian Mixture) | Probabilistic | n_components=auto |
| **DBSCAN** | Density-based | eps=6.26, min_samples=3 (auto-tuned) |

---

## 📊 Results

### GTZAN Dataset - Experiment Summary

#### Configuration
- **Total Experiments**: 60
- **Train/Test Splits**: [50/50, 60/40, 70/30, 80/20]
- **Random Seeds**: [0, 42, 1337]
- **Samples**: 1,000 tracks
- **Features**: 57

#### 🏆 Best Overall Algorithm (by Average Silhouette Score)

**Winner: DBSCAN**

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Silhouette | **0.2231** | Moderate cluster separation |
| Davies-Bouldin | 1.5211 | Good cluster cohesion |
| Calinski-Harabasz | 8.8049 | Reasonable separation |
| NMI | 0.0252 | Low label agreement |
| ARI | 0.0006 | Poor cluster overlap with true labels |
| Hungarian Accuracy | 0.1137 | 11.4% alignment |

#### 🎖️ Algorithm Rankings (Composite Scores)

| Rank | Algorithm | Composite Score | Best Use Case |
|------|-----------|----------------|---------------|
| 1 | **K-Means** | 5.9303 | General-purpose, fast |
| 2 | **Spectral** | 5.5184 | Complex cluster shapes |
| 3 | **MiniBatch K-Means** | 5.4905 | Large datasets |
| 4 | **GMM** | 4.1232 | Probabilistic assignments |
| 5 | **DBSCAN** | 1.5940 | Noise handling, arbitrary shapes |

#### 🎯 Best Single Configuration

- **Algorithm**: DBSCAN
- **Train Ratio**: 50/50
- **Random Seed**: 42
- **Silhouette Score**: **0.3460** (highest achieved)

### Generated Outputs

#### 📊 CSV Files (4)
- `clustering_results_detailed.csv` (10.12 KB)
- `clustering_results_summary.csv` (4.82 KB)
- `best_configurations.csv` (1.12 KB)
- `dbscan_detailed_results.csv` (2.58 KB)

#### � Visualizations (8)
- Algorithm Performance Ranking (192 KB)
- Algorithm Stability Analysis (196 KB)
- Boxplot All Metrics (168 KB)
- DBSCAN Analysis (352 KB)
- Distribution by Algorithm (361 KB)
- Heatmaps All Metrics (756 KB)
- Metrics Comparison (349 KB)
- Radar Chart Comparison (723 KB)

#### 📝 Reports (1)
- `experiment_report.txt` (1.38 KB)

**Total Files**: 13 files across all formats

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Jupyter Notebook

### Setup

```bash
# Clone the repository
git clone https://github.com/anisharma07/music-genere-presentation.git
cd music-genere-presentation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install audio processing libraries
pip install librosa soundfile audioread
```

### Required Libraries
```
numpy
pandas
scikit-learn
librosa
matplotlib
seaborn
jupyter
scipy
wandb  # For experiment tracking
```

---

## 🚀 Usage

### Run Individual Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open desired notebook:
# - 1.ml-gtzan-genere.ipynb (GTZAN analysis)
# - 2.FMA_Music_Genre.ipynb (FMA analysis)
# - 3.msd-notebook.ipynb (MSD analysis)
# - 4-spotify-notebook.ipynb (Spotify analysis)
```

### Batch Processing

Each notebook follows this workflow:
1. **Load Data**: Import and explore dataset
2. **Feature Extraction**: Extract audio features using Librosa
3. **Preprocessing**: Clean, normalize, and reduce dimensions
4. **Clustering**: Apply multiple algorithms
5. **Evaluation**: Calculate metrics and generate visualizations
6. **Export**: Save results to `results/` directory

---

## 📐 Evaluation Metrics

### Internal Metrics (No Labels Required)

| Metric | Range | Optimal | Description |
|--------|-------|---------|-------------|
| **Silhouette Score** | [-1, 1] | → 1 | Cluster cohesion and separation |
| **Davies-Bouldin Index** | [0, ∞] | → 0 | Average cluster similarity |
| **Calinski-Harabasz Score** | [0, ∞] | → ∞ | Between/within cluster variance ratio |

### External Metrics (With True Labels)

| Metric | Range | Optimal | Description |
|--------|-------|---------|-------------|
| **NMI** (Normalized Mutual Information) | [0, 1] | → 1 | Information shared between clusters and labels |
| **ARI** (Adjusted Rand Index) | [-1, 1] | → 1 | Similarity between cluster and true partitions |
| **Hungarian Accuracy** | [0, 1] | → 1 | Best label assignment accuracy |

---

## 🔍 Key Findings

### ✅ Successes
- **Unsupervised Clustering**: Successfully identified meaningful patterns without genre labels
- **Algorithm Comparison**: Comprehensive evaluation of 5 clustering algorithms
- **Robustness Testing**: Multiple train/test splits and random seeds ensure reliable results
- **Auto-tuning**: Automatically determined optimal parameters (e.g., DBSCAN eps/min_samples)
- **Scalability**: Methods work across datasets of varying sizes (1K-8K tracks)

### 📊 Insights
- **K-Means** performs best overall for balanced, spherical clusters
- **DBSCAN** excels at finding arbitrary-shaped clusters and handling outliers
- **Spectral Clustering** shows promise for complex genre relationships
- Genre boundaries are often **fuzzy** in audio feature space
- Feature selection significantly impacts clustering quality

### ⚠️ Challenges
- Low external metric scores indicate genre classification is inherently difficult
- Some genres (e.g., rock/metal, jazz/blues) have overlapping feature distributions
- High-dimensional feature spaces require careful dimensionality reduction
- Computational cost varies significantly across algorithms

---

## 🔮 Future Work

### Short Term
- [ ] Explore additional feature engineering techniques
- [ ] Experiment with ensemble clustering methods
- [ ] Apply deep learning-based feature extraction (CNN, RNN)
- [ ] Implement semi-supervised learning approaches

### Long Term
- [ ] Real-time genre classification system
- [ ] Multi-label genre classification (songs can have multiple genres)
- [ ] Cross-dataset validation and transfer learning
- [ ] Web application for interactive genre exploration
- [ ] Integration with music streaming platforms

### Research Directions
- [ ] Temporal analysis of genre evolution over time
- [ ] Cultural and regional music pattern discovery
- [ ] Mood and emotion-based clustering
- [ ] Hybrid supervised-unsupervised approaches

---

## 📚 References

1. GTZAN Dataset: [Marsyas](http://marsyas.info/downloads/datasets.html)
2. FMA Dataset: [Free Music Archive](https://github.com/mdeff/fma)
3. Million Song Dataset: [Columbia/LabROSA](http://millionsongdataset.com/)
4. Librosa: [Audio Analysis Library](https://librosa.org/)
5. Scikit-learn: [Machine Learning in Python](https://scikit-learn.org/)

---

## 👥 Contributors

- **Anirudh Sharma** - [anisharma07](https://github.com/anisharma07)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset providers and maintainers
- Librosa and Scikit-learn communities
- Weights & Biases for experiment tracking

---

## 📞 Contact

For questions, suggestions, or collaborations:
- **GitHub**: [@anisharma07](https://github.com/anisharma07)
- **Repository**: [music-genere-presentation](https://github.com/anisharma07/music-genere-presentation)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ and 🎵

</div>


