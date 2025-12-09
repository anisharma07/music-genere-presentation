# Dataset Labels and Genre Information

This document provides comprehensive information about the labels and genres available in each dataset used for the music clustering analysis.

## Overview

The project includes four different music datasets, each with distinct genre distributions, plus a unified combined dataset.

| Dataset | Total Samples | Number of Genres | Balance |
|---------|--------------|------------------|---------|
| **Combined** | **26,484** | **19** | **Highly Imbalanced** |
| Indian | 500 | 5 | Perfectly Balanced |
| FMA Small | 7,997 | 8 | Perfectly Balanced |
| FMA Medium | 16,988 | 16 | Highly Imbalanced |
| GTZAN | 999 | 10 | Nearly Balanced |

> **NEW:** A combined dataset merging all four sources with unified genre labels is now available! See [COMBINED_DATASET_README.md](COMBINED_DATASET_README.md) for complete documentation.

---

## 0. Combined Music Dataset (NEW!)

**Location:** `data/combined-set.csv` (features) + `data/combined-labels.csv` (labels)

### Statistics
- **Total Samples:** 26,484
- **Number of Genres:** 19 (unified labels)
- **Number of Features:** 45 (PCA components)
- **Distribution:** Highly imbalanced (real-world scenario)

### Quick Overview
The combined dataset merges all four music datasets with standardized genre labels:
- **Sources:** Indian (500) + FMA Small (7,997) + FMA Medium (16,988) + GTZAN (999)
- **Genre Unification:** 31 original labels → 19 unified genres
- **Example Mappings:** bollypop→Pop, carnatic→Classical, ghazal→World, hiphop→Hip-Hop

### Top 10 Genres
| Genre | Samples | Percentage |
|-------|---------|------------|
| Rock | 7,198 | 27.2% |
| Electronic | 6,311 | 23.8% |
| Hip-Hop | 2,297 | 8.7% |
| Experimental | 2,250 | 8.5% |
| Folk | 1,518 | 5.7% |
| Pop | 1,386 | 5.2% |
| Instrumental | 1,349 | 5.1% |
| World | 1,218 | 4.6% |
| Classical | 919 | 3.5% |
| Old-Time | 510 | 1.9% |

### Key Characteristics
- 🎵 **Large-Scale:** 26,484 samples for robust training
- 🌍 **Multi-Source:** Combines Western and Indian music traditions
- ⚠️ **Imbalanced:** Rock + Electronic = 51% of data
- 📊 **19 Unified Genres:** From Classical to Electronic to World
- 🔗 **Traceable:** Each sample tagged with source dataset

### Usage
```python
# Combined Dataset
DATA_CSV_PATH = "../data/combined-set.csv"
LABEL_CSV_PATH = "../data/combined-labels.csv"
LABEL_CSV_LABEL_COLUMN = "label"
```

📖 **Full Documentation:** See [COMBINED_DATASET_README.md](COMBINED_DATASET_README.md) for complete details on genre mapping, source contributions, and best practices.

---

## 1. Indian Music Dataset

**Location:** `results/normalization/indian_labels.csv`

### Statistics
- **Total Samples:** 500
- **Number of Genres:** 5
- **Distribution:** Perfectly balanced (20% per genre)

### Genres
The Indian dataset focuses on traditional and contemporary Indian music styles:

| Genre | Samples | Percentage | Description |
|-------|---------|------------|-------------|
| Bollypop | 100 | 20.0% | Bollywood pop music - fusion of Indian and Western pop |
| Carnatic | 100 | 20.0% | South Indian classical music |
| Ghazal | 100 | 20.0% | Poetic form of Urdu/Hindi music |
| Semiclassical | 100 | 20.0% | Light classical Indian music |
| Sufi | 100 | 20.0% | Devotional Islamic music from South Asia |

### Key Characteristics
- ✅ **Perfectly balanced:** Each genre has exactly 100 samples
- 🎯 **Specialized:** Focuses on Indian music traditions
- 🎵 **Cultural diversity:** Represents different regional and religious musical traditions

---

## 2. FMA Small Dataset

**Location:** `results/normalization/fma_small_labels.csv`

### Statistics
- **Total Samples:** 7,997
- **Number of Genres:** 8
- **Distribution:** Perfectly balanced (~12.5% per genre)

### Genres
A curated subset of the Free Music Archive with balanced representation:

| Genre | Samples | Percentage |
|-------|---------|------------|
| Electronic | 999 | 12.5% |
| Experimental | 999 | 12.5% |
| Folk | 1,000 | 12.5% |
| Hip-Hop | 1,000 | 12.5% |
| Instrumental | 1,000 | 12.5% |
| International | 1,000 | 12.5% |
| Pop | 1,000 | 12.5% |
| Rock | 999 | 12.5% |

### Key Characteristics
- ✅ **Nearly perfectly balanced:** All genres have ~1000 samples
- 🌍 **Diverse:** Covers major contemporary music genres
- 📊 **Ideal for ML:** Balanced distribution prevents bias toward any genre

---

## 3. FMA Medium Dataset

**Location:** `results/normalization/fma_medium_labels.csv`

### Statistics
- **Total Samples:** 16,988
- **Number of Genres:** 16
- **Distribution:** Highly imbalanced

### Genres
A larger subset with more genre diversity but significant class imbalance:

| Genre | Samples | Percentage | Category |
|-------|---------|------------|----------|
| Rock | 6,099 | 35.9% | 🔴 Dominant |
| Electronic | 5,312 | 31.3% | 🔴 Dominant |
| Experimental | 1,251 | 7.4% | 🟡 Medium |
| Hip-Hop | 1,197 | 7.0% | 🟡 Medium |
| Classical | 619 | 3.6% | 🟠 Minor |
| Folk | 518 | 3.0% | 🟠 Minor |
| Old-Time / Historic | 510 | 3.0% | 🟠 Minor |
| Jazz | 384 | 2.3% | 🟠 Minor |
| Instrumental | 349 | 2.1% | 🟠 Minor |
| Pop | 186 | 1.1% | 🔵 Rare |
| Country | 178 | 1.0% | 🔵 Rare |
| Soul-RnB | 154 | 0.9% | 🔵 Rare |
| Spoken | 118 | 0.7% | 🔵 Rare |
| Blues | 74 | 0.4% | 🔵 Rare |
| Easy Listening | 21 | 0.1% | ⚪ Very Rare |
| International | 18 | 0.1% | ⚪ Very Rare |

### Key Characteristics
- ⚠️ **Highly imbalanced:** Rock and Electronic dominate with 67.2% of data
- 🎵 **Diverse:** 16 different genres provide rich variety
- 📊 **Challenging for ML:** Class imbalance requires careful handling
- 🌐 **Comprehensive:** Includes both mainstream and niche genres

---

## 4. GTZAN Dataset

**Location:** `results/normalization/gtzan_labels.csv`

### Statistics
- **Total Samples:** 999
- **Number of Genres:** 10
- **Distribution:** Nearly perfectly balanced (~10% per genre)

### Genres
The GTZAN Genre Collection - a benchmark dataset for music genre classification:

| Genre | Samples | Percentage |
|-------|---------|------------|
| Blues | 100 | 10.0% |
| Classical | 100 | 10.0% |
| Country | 100 | 10.0% |
| Disco | 100 | 10.0% |
| Hiphop | 100 | 10.0% |
| Jazz | 99 | 9.9% |
| Metal | 100 | 10.0% |
| Pop | 100 | 10.0% |
| Reggae | 100 | 10.0% |
| Rock | 100 | 10.0% |

### Key Characteristics
- ✅ **Nearly perfectly balanced:** Each genre has 100 samples (except Jazz with 99)
- 🎯 **Benchmark dataset:** Widely used in music information retrieval research
- 🎵 **Diverse styles:** Covers a wide range of Western popular music genres
- 📊 **Ideal for research:** Small, balanced, and well-documented

---

## Dataset Comparison

### Balance Comparison
```
Indian:      ████████████████████ (Perfectly Balanced)
FMA Small:   ████████████████████ (Perfectly Balanced)
GTZAN:       ████████████████████ (Nearly Balanced)
FMA Medium:  ████░░░░░░░░░░░░░░░░ (Highly Imbalanced)
```

### Size Comparison
```
FMA Medium:  ████████████████████ (16,988 samples)
FMA Small:   ████████░░░░░░░░░░░░ (7,997 samples)
GTZAN:       █░░░░░░░░░░░░░░░░░░░ (999 samples)
Indian:      █░░░░░░░░░░░░░░░░░░░ (500 samples)
```

### Genre Diversity
```
FMA Medium:  ████████████████     (16 genres)
GTZAN:       ██████████           (10 genres)
FMA Small:   ████████             (8 genres)
Indian:      █████                (5 genres)
```

---

## Usage in Clustering Analysis

### Label File Paths
The label files are located in the `results/normalization/` directory (individual datasets) or `data/` directory (combined dataset) and can be configured in the clustering notebook:

```python
# Combined Dataset (RECOMMENDED FOR LARGE-SCALE EXPERIMENTS)
DATA_CSV_PATH = "../data/combined-set.csv"
LABEL_CSV_PATH = "../data/combined-labels.csv"

# Indian Dataset
DATA_CSV_PATH = "../results/pca/indian_pca.csv"
LABEL_CSV_PATH = "../results/normalization/indian_labels.csv"

# FMA Small Dataset
DATA_CSV_PATH = "../results/pca/fma_small_pca.csv"
LABEL_CSV_PATH = "../results/normalization/fma_small_labels.csv"

# FMA Medium Dataset
DATA_CSV_PATH = "../results/pca/fma_medium_pca.csv"
LABEL_CSV_PATH = "../results/normalization/fma_medium_labels.csv"

# GTZAN Dataset
DATA_CSV_PATH = "../results/pca/gtzan_pca.csv"
LABEL_CSV_PATH = "../results/normalization/gtzan_labels.csv"
```

### Evaluation Metrics
When ground truth labels are available, the following external metrics are computed:

- **NRI (Normalized Rand Index):** Measures similarity between clusterings [0, 1]
- **ARI (Adjusted Rand Index):** Adjusted-for-chance version [-1, 1]
- **Purity:** Percentage of correctly clustered samples [0, 1]

### Recommendations by Dataset

#### Combined Dataset (NEW!)
- ✅ Best for: Large-scale experiments, cross-dataset generalization
- ✅ Optimal k values: 19 (matches unified genres), also try 10, 15, 25
- ⚠️ Challenge: Highly imbalanced (requires special handling)
- 💡 Strength: 26,484 samples from diverse sources

#### Indian Dataset
- ✅ Best for: Specialized Indian music analysis
- ✅ Optimal k values: 5 (matches true genres)
- ⚠️ Note: Smaller dataset may require careful validation

#### FMA Small
- ✅ Best for: General music genre clustering research
- ✅ Optimal k values: 8 (matches true genres)
- ✅ Advantage: Perfect balance reduces bias

#### FMA Medium
- ✅ Best for: Real-world scenarios with imbalanced data
- ⚠️ Optimal k values: 16 (but consider class imbalance)
- ⚠️ Challenge: Requires techniques to handle imbalanced classes

#### GTZAN
- ✅ Best for: Benchmark comparisons and reproducible research
- ✅ Optimal k values: 10 (matches true genres)
- ✅ Advantage: Well-documented and widely recognized

---

## Data Quality Notes

### Label Consistency
All label files follow the same format:
- Single column named `label`
- One label per row
- No missing values
- UTF-8 encoding

### Alignment with Feature Files
The label files are aligned with their corresponding PCA-reduced feature files:
- Row order matches exactly
- Number of samples matches
- No index column needed

---

## References

1. **FMA Dataset:** Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). FMA: A dataset for music analysis. In 18th International Society for Music Information Retrieval Conference (ISMIR).

2. **GTZAN Dataset:** Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. IEEE Transactions on speech and audio processing, 10(5), 293-302.

3. **Indian Music Dataset:** Custom curated collection focusing on traditional and contemporary Indian music styles.

---

## Last Updated
December 5, 2025

---

## Contact
For questions about the datasets or label information, please refer to the main project documentation or contact the project maintainers.
