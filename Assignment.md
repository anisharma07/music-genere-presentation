# 🎵 Unsupervised Music Genre Discovery Using Audio Feature Learning

## 📘 Project Overview
To Develop a machine learning project on **Unsupervised Music Genre Discovery Using Audio Feature Learning** to cluster and analyze musical genres using extracted audio features from multiple datasets.

---

## 🎯 Objectives
- Apply **unsupervised learning algorithms** to discover hidden genre clusters.  
- Perform **comprehensive dataset analysis**, feature extraction, and model evaluation.  
- Compare algorithms across different datasets and splits using multiple metrics.

---

## 🗂️ Datasets
Use at least **four different datasets** from diverse sources:
1. **GTZAN Dataset** – 10 genres, 1000 tracks  
2. **FMA-Small Dataset** – 8,000 tracks across 8 genres  
3. **Million Song Dataset** – Regional music data  
4. **Spotify Tracks Dataset** – Audio features via Spotify API  

---

## 🧹 Data Analysis & Cleaning
Perform:
- Descriptive Statistics (mean, median, mode, quartiles, IQR)  
- Identify & remove outliers (boxplots)  
- Handle missing values (replace with mean)  
- Correlation analysis of key features  
- Distribution and skewness analysis  
- Compute trimmed mean, standard deviation, and interpret data quality  

---

## 🧠 Methodology
### Feature Extraction (Librosa)
- MFCC (20–40 coefficients)  
- Chroma Features  
- Spectral Centroid, Bandwidth, Roll-off  
- Zero Crossing Rate  

### Dimensionality Reduction
- **PCA** (retain 95% variance)  
- **t-SNE** or **UMAP** for visualization  

### Clustering Algorithms
- K-Means  
- Gaussian Mixture Models (GMM)  
- DBSCAN  
- Hierarchical Clustering  
- Autoencoder + K-Means (Advanced)

---

## ⚙️ Experimental Setup
- Data splits: 50–50, 60–40, 70–30, 80–20  
- Cross-validation: 5-fold  
- Randomized experiments for consistency  

---

## 📊 Evaluation Metrics
Use at least **6 evaluation metrics**:
1. Silhouette Score  
2. Davies–Bouldin Index  
3. Calinski–Harabasz Index  
4. Adjusted Rand Index (ARI)  
5. Mutual Information (MI)  
6. Cluster Purity / Homogeneity  

---

## 🧩 Deliverables
- 📄 **IEEE-format Report**  
  - Abstract, Introduction, Related Work, Implementation, Results, Conclusion  
- 🖼️ **PPT Presentation**  
- 💻 **Source Code + Dataset Links** (GitHub / Kaggle / W&B)

---

## 🛠️ Tools & Environment
- Python, Jupyter/Colab  
- Librosa, Scikit-learn, NumPy, Pandas  
- Matplotlib, Seaborn, Plotly  
- Weights & Biases (W&B) for experiment tracking  

---

## 📚 Reference Papers
1. Tzanetakis & Cook, *IEEE Trans. Speech and Audio Processing*, 2002  
2. Humphrey et al., *IEEE Trans. Multimedia*, 2013  
3. Choi et al., *ISMIR 2017*  
4. Pons et al., *IEEE/ACM Trans. Audio, Speech, and Language Processing*, 2018  
5. Gururani et al., *ICASSP 2020*  

---

## 🧾 Report Structure (IEEE Format)
- Abstract (~500 words)  
- Keywords (5 words)  
- Introduction  
- Related Work  
- Implementation  
- Theoretical/Mathematical Analysis  
- Results & Discussion  
- Conclusion  
- References  
- Acknowledgement  
- Research Data & Code Links  
- Brief Biodata  

---

> 💡 **Goal:** Achieve meaningful unsupervised clustering of music genres using feature learning techniques, analyze multiple datasets, and present comprehensive quantitative and visual results.
