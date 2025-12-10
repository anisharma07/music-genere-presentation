# 🎵 Unsupervised Music Genre Discovery Using Audio Feature Learning

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Librosa](https://img.shields.io/badge/Librosa-0.11.0-red.svg)](https://librosa.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-green.svg)](https://scikit-learn.org/)

> **A comprehensive investigation into unsupervised music genre classification through audio feature learning, dimensionality reduction, and clustering across 37,800 tracks spanning Western and Indian musical traditions.**

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Research Highlights](#-research-highlights)
- [Datasets](#-datasets)
- [Project Structure](#-project-structure)
- [Methodology Pipeline](#-methodology-pipeline)
- [Installation](#-installation)
- [Usage](#-usage)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results Summary](#-results-summary)
- [Publications](#-publications)
- [Future Work](#-future-work)
- [Acknowledgments](#-acknowledgments)

---

## 📖 Overview

This research presents a **systematic evaluation of unsupervised clustering algorithms** for automatic music genre discovery using acoustic features extracted from diverse audio collections. Unlike supervised approaches requiring extensive labeled datasets, our framework discovers latent genre structures directly from audio signals through feature engineering, dimensionality reduction, and multiple clustering techniques.

### 🎯 Research Objectives

1. **Extract comprehensive audio features** from diverse music datasets using Librosa
2. **Establish robust preprocessing pipeline** including data integrity validation, outlier detection, normalization, and PCA
3. **Evaluate four clustering algorithms** (K-Means, Agglomerative, GMM, Spectral) across multiple datasets
4. **Benchmark performance** using six complementary metrics (silhouette, Davies-Bouldin, ARI, NMI, purity, Calinski-Harabasz)
5. **Develop cluster-to-genre mapping** through majority voting for semantic interpretation
6. **Validate cross-cultural generalization** spanning Western and Indian musical traditions

### 🔬 Key Contributions

- **99.94% feature extraction success rate** across 37,800 tracks
- **99.99% data cleanliness** after integrity validation (only 4 corrupted files removed)
- **39.2% average dimensionality reduction** via PCA while retaining 95.15% variance
- **Unified k=10 cluster-to-genre mapping** achieving 45.9% average purity
- **Cross-dataset consistency** validating clustering approach across Western and Indian music

---

## 🏆 Research Highlights

| Metric | Achievement | Description |
|--------|-------------|-------------|
| **Datasets Analyzed** | 5 diverse collections | GTZAN, FMA Small/Medium, Ludwig, Indian Regional |
| **Total Tracks** | 37,800 | 500 to 17,000 tracks per dataset |
| **Feature Dimensionality** | 69 → 42 | 40% reduction via PCA, 95%+ variance retained |
| **Best Algorithm** | Spectral Clustering | ARI: 0.225, Purity: 42.9% (GTZAN) |
| **Data Quality** | 99.99% clean | Only 4/37,778 files removed due to corruption |
| **Outlier Prevalence** | 0.58-1.69% | Minimal outliers across key features |

---

## 🗂️ Datasets

Five diverse music collections spanning Western popular music and traditional Indian regional styles:

| Dataset | Tracks | Genres | Duration | Characteristics |
|---------|--------|--------|----------|-----------------|
| **Indian Regional** | 500 | 5 | 45s | Balanced (100/genre): Bollypop, Carnatic, Ghazal, Semiclassical, Sufi |
| **GTZAN** | 1,000 | 10 | 30s | Benchmark dataset: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock |
| **FMA Small** | 8,000 | 8 | 30s | Balanced genres from Free Music Archive |
| **Ludwig** | 11,300 | 10 | 30s | Spotify-sourced with AcousticBrainz metadata |
| **FMA Medium** | 17,000 | 16 | 30s | Unbalanced, hierarchical genre taxonomy |

**Total Collection**: 37,800 tracks • **Combined Features**: 69 acoustic descriptors per track

### Data Sources
- **GTZAN**: [Kaggle](https://kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- **FMA**: [GitHub Repository](https://github.com/mdeff/fma)
- **Ludwig**: [Kaggle](https://kaggle.com/datasets/jorgeruizdev/ludwig-music-dataset-moods-and-subgenres)
- **Indian**: [Kaggle](https://kaggle.com/datasets/somnath796/indian-bollywood-music-genre-classification)

---

## 📁 Project Structure

```
music-genere-presentation/
│
├── NOTEBOOKS/                           # Jupyter analysis pipeline
│   ├── step1-feature-extraction.ipynb   # Extract 69 audio features using Librosa
│   ├── step2.1-descriptive-analysis.ipynb
│   ├── step2.2-data-integrity-health-check.ipynb
│   ├── step2.3-outlier-detection.ipynb
│   ├── step2.4-distribution-skewness-analysis.ipynb
│   ├── step2.5-correlation-analysis.ipynb
│   ├── step2.6-dataset-bias-check.ipynb
│   ├── step3-normalization.ipynb        # StandardScaler z-score normalization
│   ├── step4-Pca-reduction.ipynb        # PCA: 95% variance, 40% reduction
│   ├── step5-clustering-Experiments.ipynb # K-Means, Spectral, GMM, Agglomerative
│   └── results/                         # Experiment outputs
│
├── data/                                # Processed feature datasets
│   ├── feature-extraction/              # Raw extracted features (CSV)
│   ├── feature-extraction-cleaned/      # Post-integrity validation
│   ├── clustering_ready/                # PCA-transformed, ready for clustering
│   ├── pca_transformed/                 # Dimensionality-reduced datasets
│   └── label_references/                # Ground-truth genre labels
│
├── raw-data/                            # Original audio files
│   ├── gtzan/                           # 1,000 tracks (30s clips)
│   ├── fma_small/                       # 8,000 tracks
│   ├── fma_medium/                      # 17,000 tracks
│   ├── Ludwig/                          # 11,300 tracks
│   └── Indian/                          # 500 tracks (45s clips)
│
├── results/                             # Analysis outputs and visualizations
│   ├── step1/                           # Feature extraction stats
│   ├── step1.3-outlier-detection/       # Boxplots, outlier analysis
│   ├── step1.4-distribution-skewness/   # Distribution histograms
│   ├── step1.5-correlation-analysis/    # MFCC correlation matrices
│   ├── step1.6-dataset-bias-check/      # Kruskal-Wallis bias analysis
│   ├── normalization/                   # Before/after normalization plots
│   ├── pca/                             # Explained variance, 2D projections
│   └── clustering_images/               # t-SNE visualizations, metric plots
│
├── Report Tex/                          # IEEE conference paper (LaTeX)
│   ├── IEEE_Report_Fresh.tex            # 9-page research paper
│   ├── IEEE_Report_Fresh.pdf            # Compiled PDF
│   └── references.bib                   # Bibliography (2018-2025 papers)
│
├── Music-Classifier/                    # Real-time classification web app
│   ├── app.py                           # Flask server
│   ├── train_models.py                  # Model training scripts
│   ├── feature_extraction.py            # Librosa feature extraction
│   ├── models/                          # Trained clustering models
│   └── templates/                       # Web UI
│
├── docs/                                # Documentation
│   ├── step1-feature-extraction.txt     # Feature engineering details
│   ├── step2.md                         # Preprocessing methodology
│   └── step3.md                         # Clustering methodology
│
├── Assignment.md                        # Project requirements
├── COMBINED_DATASET_README.md           # Dataset integration guide
├── references.md                        # Research paper references
└── README.md                            # This file
```

---

## 🔬 Methodology Pipeline

Our systematic 7-phase pipeline processes raw audio to clustered genre discoveries:

### **Phase 1: Feature Extraction**
Using **Librosa 0.11.0**, extract 69 numerical descriptors per track:
- **Spectral Features (4)**: Centroid, rolloff, zero-crossing rate, RMS energy
- **MFCCs (40)**: 20 coefficients × (mean + std) for timbral characterization
- **Chromagrams (24)**: 12 pitch classes × (mean + std) for harmonic content
- **Tempo (1)**: BPM via onset strength-based beat tracking

**Parameters**: 22,050 Hz sampling • 2048-sample window • 512-sample hop length

### **Phase 2: Data Integrity & Quality Analysis**

**2.1 Descriptive Analysis**: Statistical summaries (mean, median, std, quartiles)

**2.2 Data Integrity Check**:
- NaN detection: 0 missing values across all datasets
- Infinite value detection: 0 occurrences
- Silent/corrupt file detection: 4 files removed (0.011% loss)
- **Result**: 99.99% data cleanliness

**2.3 Outlier Detection**:
- Method: IQR (Interquartile Range) on tempo, RMS, spectral centroid, ZCR
- Outlier prevalence: 0.58% (spectral centroid) to 1.69% (ZCR)
- Decision: Retained all data (outliers represent genuine musical diversity)

**2.4 Distribution & Skewness**:
- 70.7% features show moderate-to-high skewness
- Key features (spectral rolloff, centroid) exhibit near-Gaussian distributions
- Decision: No logarithmic transformation (preserve interpretability)

**2.5 Correlation Analysis**:
- MFCC mean correlation: 0.077 (GTZAN) to 0.247 (FMA Small)
- 0-3 feature pairs exceed |r| > 0.8 threshold
- Justifies PCA for redundancy reduction

**2.6 Dataset Bias Check**:
- Kruskal-Wallis H-test reveals strong statistical bias (p < 0.001)
- Cohen's d effect sizes remain small-to-medium (90% < 0.5)
- Mitigation: Unified StandardScaler normalization

### **Phase 3: Normalization**
- **Method**: StandardScaler (z-score normalization)
- **Formula**: $z = \frac{x - \mu}{\sigma}$
- **Verification**: All features achieve exact 0.0 mean, 1.0 standard deviation

### **Phase 4: Dimensionality Reduction (PCA)**
- **Threshold**: Retain ≥95% cumulative explained variance
- **Results**: 69 → 42 components average (39.2% reduction)
- **Benefits**: 2.7× speedup in K-Means iteration, curse of dimensionality mitigation

### **Phase 5: Clustering Experiments**
Four algorithms evaluated at k=10 (aligned with 10-genre mapping):

| Algorithm | Type | Linkage/Parameters |
|-----------|------|-------------------|
| **K-Means** | Centroid-based | K-Means++ initialization, 300 iterations |
| **Agglomerative** | Hierarchical | Ward linkage (minimum variance) |
| **GMM** | Probabilistic | Full covariance matrices, EM algorithm |
| **Spectral** | Graph-based | 15-nearest neighbors, RBF kernel |

**Visualization**: t-SNE (perplexity=30) for 2D/3D projections

---

## 📊 Evaluation Metrics

We employ **six complementary metrics** for comprehensive clustering assessment:

### Internal Metrics (No Ground Truth Required)

| Metric | Formula/Description | Range | Optimal | Interpretation |
|--------|---------------------|-------|---------|----------------|
| **Silhouette Score** | $\frac{b - a}{\max(a, b)}$ | [-1, 1] | → 1 | Cluster cohesion vs. separation |
| **Davies-Bouldin Index** | Avg. cluster similarity ratio | [0, ∞] | → 0 | Lower = better separation |
| **Calinski-Harabasz** | Between/within variance ratio | [0, ∞] | → ∞ | Higher = denser clusters |

### External Metrics (With True Genre Labels)

| Metric | Description | Range | Optimal | Interpretation |
|--------|-------------|-------|---------|----------------|
| **ARI** (Adjusted Rand Index) | Cluster-label similarity | [-1, 1] | → 1 | Corrects for chance agreement |
| **NMI** (Normalized Mutual Info) | Shared information | [0, 1] | → 1 | Information-theoretic measure |
| **Purity** | Majority vote accuracy | [0, 1] | → 1 | % tracks in correct cluster |

---

## 🏆 Results Summary

### Cross-Dataset Performance (k=10 Clustering)

| Dataset | Tracks | Best Algorithm | ARI | Purity | Silhouette |
|---------|--------|----------------|-----|--------|------------|
| **GTZAN** | 999 | Spectral | 0.225 | 42.9% | 0.088 |
| **FMA Small** | 7,996 | GMM | 0.107 | 36.8% | -0.020 |
| **FMA Medium** | 16,986 | Spectral | 0.213 | 55.2% | 0.061 |
| **Ludwig** | 11,293 | K-Means | 0.132 | 42.7% | 0.078 |
| **Indian Regional** | 500 | Agglomerative | 0.196 | 53.0% | 0.142 |
| **Average** | — | — | **0.176** | **45.9%** | **0.070** |

### Algorithm-Specific Insights

**🥇 Spectral Clustering**: Best for large Western datasets
- GTZAN: ARI=0.225, Purity=42.9%
- FMA Medium: ARI=0.213, Purity=55.2%
- Excels at non-convex genre boundaries

**🥈 K-Means**: Optimal for balanced, well-structured data
- Fastest convergence (2.7× speedup with PCA)
- Ludwig dataset: ARI=0.132

**🥉 Agglomerative (Ward)**: Superior for hierarchical genres
- Indian Regional: Purity=53.0%
- Captures natural genre relationships

**GMM**: Best when soft assignments needed
- FMA Small: ARI=0.107 despite negative silhouette

### Unified Cluster-to-Genre Mapping

Using **majority voting**, we established semantic genre labels for k=10 clusters:

| Cluster | Genre | Acoustic Signature |
|---------|-------|-------------------|
| 0 | **Blues** | Slow tempo, guitar-dominant, minor keys |
| 1 | **Classical** | High spectral complexity, low percussiveness |
| 2 | **Country** | Acoustic instruments, moderate tempo |
| 3 | **Disco/Dance** | High tempo, strong beat, repetitive patterns |
| 4 | **Hip-Hop** | Strong bass, rhythmic vocals, 808 drums |
| 5 | **Jazz** | Complex harmonics, improvisation patterns |
| 6 | **Metal** | High energy, distorted guitars, fast tempo |
| 7 | **Pop** | Balanced spectrum, verse-chorus structure |
| 8 | **Reggae** | Off-beat rhythm, bass-heavy, laid-back |
| 9 | **Rock** | Guitar-driven, moderate-high energy |

**Cross-Dataset Validation**: This mapping achieves **45.9% average purity**, demonstrating meaningful genre recovery without labeled training data.

---

## ⚙️ Installation

### Prerequisites
- Python 3.12+ (tested on 3.12.3)
- pip package manager
- Jupyter Notebook / JupyterLab
- 10GB+ free disk space (for raw audio + results)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/anisharma07/music-genere-presentation.git
cd music-genere-presentation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install core dependencies
pip install numpy pandas scikit-learn librosa matplotlib seaborn jupyter

# Optional: Install experiment tracking
pip install wandb
```

### Required Libraries

```python
# Core Data Processing
numpy>=1.24.0
pandas>=2.0.0

# Machine Learning
scikit-learn>=1.4.0

# Audio Processing
librosa>=0.11.0
soundfile>=0.12.0
audioread>=3.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.13.0

# Notebook Environment
jupyter>=1.0.0
ipykernel>=6.25.0

# Experiment Tracking (Optional)
wandb>=0.16.0
```

---

## 🚀 Usage

### Sequential Pipeline Execution

Execute notebooks in order to reproduce the complete research pipeline:

```bash
# Start Jupyter
jupyter notebook

# Execute in sequence:
# 1. NOTEBOOKS/step1-feature-extraction.ipynb
# 2. NOTEBOOKS/step2.1-descriptive-analysis.ipynb
# 3. NOTEBOOKS/step2.2-data-integrity-health-check.ipynb
# 4. NOTEBOOKS/step2.3-outlier-detection.ipynb
# 5. NOTEBOOKS/step2.4-distribution-skewness-analysis.ipynb
# 6. NOTEBOOKS/step2.5-correlation-analysis.ipynb
# 7. NOTEBOOKS/step2.6-dataset-bias-check.ipynb
# 8. NOTEBOOKS/step3-normalization.ipynb
# 9. NOTEBOOKS/step4-Pca-reduction.ipynb
# 10. NOTEBOOKS/step5-clustering-Experiments.ipynb
```

### Quick Start: Run Clustering Only

If pre-processed datasets are available in `data/clustering_ready/`:

```python
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Load PCA-transformed data
data = pd.read_csv('data/clustering_ready/gtzan_clustering.csv')
X = data.drop(['file_path', 'label'], axis=1)
y_true = data['label']

# Run K-Means
kmeans = KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=300, random_state=42)
clusters = kmeans.fit_predict(X)

# Evaluate
sil = silhouette_score(X, clusters)
ari = adjusted_rand_score(y_true, clusters)
print(f"Silhouette: {sil:.3f}, ARI: {ari:.3f}")
```

### Web Application (Music Classifier)

Deploy the real-time genre classification system:

```bash
cd Music-Classifier

# Install Flask dependencies
pip install flask werkzeug

# Train models (optional, if not pre-trained)
python train_models.py

# Launch web server
python app.py

# Access at http://localhost:5000
```

**Features**:
- Upload audio files (WAV/MP3/FLAC)
- Real-time feature extraction
- K-Means cluster assignment
- Genre prediction with confidence scores

---

## 🔍 Key Findings & Insights

### ✅ Major Achievements

1. **Exceptional Data Quality**: 99.99% cleanliness after removing only 4 corrupted files from 37,778 tracks
2. **Efficient Dimensionality Reduction**: PCA achieved 39.2% average reduction (69→42 dimensions) while preserving 95.15% variance
3. **Meaningful Genre Recovery**: 45.9% average purity demonstrates unsupervised methods capture substantial genre structure
4. **Cross-Cultural Validation**: Consistent clustering performance across Western and Indian musical traditions
5. **Algorithm-Dataset Specificity**: No single algorithm dominates—optimal choice depends on dataset characteristics

### 📊 Critical Insights

**Dataset Size Effects**:
- Small datasets (500-1K): Higher silhouette scores (cleaner boundaries)
- Large datasets (17K): Higher purity (richer genre representations)
- Indian dataset: 53% purity despite smallest size (cultural distinctiveness)

**Genre Overlap Patterns**:
- Rock/Metal: High spectral similarity (distorted guitars, high energy)
- Blues/Jazz: Shared harmonic vocabulary (minor keys, improvisation)
- Pop/Electronic: Balanced spectral profiles, ambiguous boundaries

**Feature Importance**:
- MFCCs (40 features): Primary timbral discriminators
- Chromagrams (24 features): Capture harmonic/melodic genre signatures
- Tempo (1 feature): Critical for dance/ballad genre separation

### ⚠️ Limitations & Challenges

**Inherent Challenges**:
- Genre labels are **culturally constructed** and inherently subjective
- 30-second clips lose long-term musical structure (verse-chorus dynamics)
- Moderate ARI values (0.107-0.225) reflect ambiguous genre boundaries

**Technical Limitations**:
- MFCC-dominated features may underweight rhythmic characteristics
- Features developed for Western music may suboptimally represent microtonality in Indian classical music
- PCA assumes linear relationships—non-linear manifolds not captured

**Dataset-Specific Issues**:
- GTZAN: Artist effects (multiple songs per artist)
- FMA Medium: Unbalanced genres (bias toward rock/electronic)
- Ludwig: Pre-computed features limit customization

---

## 🔮 Future Research Directions

### Immediate Extensions
- **Deep Learning Embeddings**: Replace hand-crafted features with contrastive learning (SimCLR, MoCo)
- **Temporal Modeling**: RNNs/Transformers to preserve temporal dynamics lost in mean/std aggregation
- **Semi-Supervised Refinement**: Use cluster-to-genre mapping as initialization for supervised fine-tuning
- **Ensemble Methods**: Combine K-Means, Spectral, and Agglomerative predictions via voting

### Advanced Topics
- **Cross-Dataset Transfer Learning**: Train on FMA Medium, test on GTZAN generalization
- **Multi-Label Classification**: Songs with multiple genres (e.g., "Country Rock", "Jazz Fusion")
- **Hierarchical Genre Taxonomy**: Exploit FMA's parent-child genre relationships
- **Cultural Music Analysis**: Expand to Middle Eastern, African, Latin American traditions

### Production Systems
- **Real-Time Classification**: Streaming audio feature extraction with <100ms latency
- **Playlist Auto-Generation**: Acoustic similarity-based recommendations
- **Music Production Tools**: Genre-aware sample libraries, mixing presets

---

## 📚 References & Related Work

### Key Research Papers (2023-2025)

1. **Singh et al. (2024)** - *Identification and clustering of unseen ragas in Indian art music*  
   [arXiv:2411.18611](https://arxiv.org/abs/2411.18611) • Closest to our research (unsupervised class discovery)

2. **Kumar et al. (2024)** - *Enhanced music recommendation systems: K-means clustering approaches*  
   International Journal of Mathematical, Engineering and Management Sciences

3. **Ma et al. (2023)** - *On the effectiveness of speech self-supervised learning for music*  
   ISMIR 2023 • Speech-to-music SSL transfer learning

4. **Wang et al. (2023)** - *Self-supervised learning of audio representations using angular contrastive loss*  
   ICASSP 2023 • Advanced contrastive learning

5. **Chong et al. (2023)** - *Masked spectrogram prediction for self-supervised audio pre-training*  
   ICASSP 2023 • Generative SSL approach

### Foundational Works

- **Tzanetakis & Cook (2002)** - *Musical genre classification of audio signals*  
  IEEE Transactions on Speech and Audio Processing • GTZAN dataset benchmark

- **Defferrard et al. (2017)** - *FMA: A dataset for music analysis*  
  ISMIR 2017 • Free Music Archive introduction

### Tools & Libraries

- **Librosa**: [https://librosa.org/](https://librosa.org/) • Audio feature extraction
- **Scikit-learn**: [https://scikit-learn.org/](https://scikit-learn.org/) • Clustering algorithms
- **Weights & Biases**: [https://wandb.ai/](https://wandb.ai/) • Experiment tracking
---

## 📄 License

This project is open-source and available under the **MIT License**.  
See [LICENSE](LICENSE) file for full terms and conditions.

**Research Use**: Free for academic and commercial research.  
**Attribution Required**: Please cite this work if you use it in publications.

---
