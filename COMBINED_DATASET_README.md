# Combined Music Dataset Documentation

This document provides comprehensive information about the combined music dataset created by merging four different music datasets with unified genre labels.

## Overview

The combined dataset integrates all available music datasets with standardized genre labels, creating a large-scale music classification dataset suitable for machine learning research.

### Quick Stats
- **Total Samples:** 26,484
- **Number of Features:** 45 (PCA components)
- **Number of Genres:** 19 (unified labels)
- **Source Datasets:** 4 (Indian, FMA Small, FMA Medium, GTZAN)
- **Balance:** Highly imbalanced (real-world distribution)

---

## File Locations

### Combined Dataset Files
- **Features:** `data/combined-set.csv` - 26,484 samples × 45 PCA components
- **Labels:** `data/combined-labels.csv` - Genre labels + source dataset information

### Usage in Clustering Analysis
```python
# Load combined dataset
DATA_CSV_PATH = "../data/combined-set.csv"
LABEL_CSV_PATH = "../data/combined-labels.csv"
LABEL_CSV_LABEL_COLUMN = "label"
```

---

## Genre Unification Strategy

### Label Mapping Rules

The following mapping strategy was used to merge similar genres across different datasets:

| Original Label | Unified Label | Source Dataset(s) |
|----------------|---------------|-------------------|
| blues, Blues | Blues | GTZAN, FMA Medium |
| classical, Classical, carnatic, semiclassical | Classical | GTZAN, FMA Medium, Indian |
| country, Country | Country | GTZAN, FMA Medium |
| disco | Disco | GTZAN |
| Electronic | Electronic | FMA Small, FMA Medium |
| Experimental | Experimental | FMA Small, FMA Medium |
| Folk | Folk | FMA Small, FMA Medium |
| hiphop, Hip-Hop | Hip-Hop | GTZAN, FMA Small, FMA Medium |
| Instrumental | Instrumental | FMA Small, FMA Medium |
| International, ghazal, sufi | World | FMA Small, FMA Medium, Indian |
| jazz, Jazz | Jazz | GTZAN, FMA Medium |
| metal | Metal | GTZAN |
| Old-Time / Historic | Old-Time | FMA Medium |
| pop, Pop, bollypop | Pop | GTZAN, FMA Small, FMA Medium, Indian |
| reggae | Reggae | GTZAN |
| rock, Rock | Rock | GTZAN, FMA Small, FMA Medium |
| Soul-RnB | Soul-RnB | FMA Medium |
| Spoken | Spoken | FMA Medium |
| Easy Listening | Easy Listening | FMA Medium |

### Key Unification Decisions

1. **Classical Music:** Merged Western classical (GTZAN, FMA) with Indian classical forms (Carnatic, Semiclassical)
2. **World Music:** Combined International music with traditional Indian genres (Ghazal, Sufi) into "World" category
3. **Pop Music:** Unified all pop variations including Bollywood pop (Bollypop)
4. **Hip-Hop:** Standardized case variations (hiphop → Hip-Hop)
5. **Blues:** Merged case variations (blues → Blues)

---

## Genre Distribution

### Complete Distribution Table

| Genre | Samples | Percentage | Category |
|-------|---------|------------|----------|
| Rock | 7,198 | 27.2% | 🔴 Dominant |
| Electronic | 6,311 | 23.8% | 🔴 Dominant |
| Hip-Hop | 2,297 | 8.7% | 🟡 Medium |
| Experimental | 2,250 | 8.5% | 🟡 Medium |
| Folk | 1,518 | 5.7% | 🟡 Medium |
| Pop | 1,386 | 5.2% | 🟡 Medium |
| Instrumental | 1,349 | 5.1% | 🟡 Medium |
| World | 1,218 | 4.6% | 🟠 Minor |
| Classical | 919 | 3.5% | 🟠 Minor |
| Old-Time | 510 | 1.9% | 🟠 Minor |
| Jazz | 483 | 1.8% | 🟠 Minor |
| Country | 278 | 1.0% | 🔵 Rare |
| Blues | 174 | 0.7% | 🔵 Rare |
| Soul-RnB | 154 | 0.6% | 🔵 Rare |
| Spoken | 118 | 0.4% | 🔵 Rare |
| Disco | 100 | 0.4% | 🔵 Rare |
| Metal | 100 | 0.4% | 🔵 Rare |
| Reggae | 100 | 0.4% | 🔵 Rare |
| Easy Listening | 21 | 0.1% | ⚪ Very Rare |

### Visual Distribution

```
Rock         ████████████████████████████ (27.2%)
Electronic   ████████████████████████     (23.8%)
Hip-Hop      █████████                    (8.7%)
Experimental █████████                    (8.5%)
Folk         ██████                       (5.7%)
Pop          █████                        (5.2%)
Instrumental █████                        (5.1%)
World        █████                        (4.6%)
Classical    ████                         (3.5%)
Old-Time     ██                           (1.9%)
Jazz         ██                           (1.8%)
Country      █                            (1.0%)
Blues        █                            (0.7%)
Soul-RnB     █                            (0.6%)
Others       █                            (1.6%)
```

### Distribution by Category

- **Dominant Genres (>20%):** Rock, Electronic — 51.0% of data
- **Medium Genres (5-10%):** Hip-Hop, Experimental, Folk, Pop, Instrumental — 32.7% of data
- **Minor Genres (1-5%):** World, Classical, Old-Time, Jazz — 11.8% of data
- **Rare Genres (<1%):** Country, Blues, Soul-RnB, Spoken, Disco, Metal, Reggae, Easy Listening — 4.5% of data

---

## Source Dataset Contributions

### Breakdown by Source

| Source Dataset | Samples | Percentage | Features Used |
|----------------|---------|------------|---------------|
| FMA Medium | 16,988 | 64.1% | 45 PCs (original) |
| FMA Small | 7,997 | 30.2% | 45 PCs (original) |
| GTZAN | 999 | 3.8% | 39 PCs → 45 PCs (padded) |
| Indian | 500 | 1.9% | 40 PCs → 45 PCs (padded) |

### Feature Padding Strategy

Since datasets had different numbers of PCA components, zero-padding was applied to create uniform feature vectors:

- **FMA Small/Medium:** 45 features (no padding needed)
- **Indian Dataset:** 40 → 45 features (5 zeros appended)
- **GTZAN Dataset:** 39 → 45 features (6 zeros appended)

This ensures all samples have identical dimensionality while preserving original information.

---

## Genre Composition by Source

### How Each Genre is Formed

#### Rock (7,198 samples)
- FMA Medium: 6,099 samples (84.7%)
- FMA Small: 999 samples (13.9%)
- GTZAN: 100 samples (1.4%)

#### Electronic (6,311 samples)
- FMA Medium: 5,312 samples (84.2%)
- FMA Small: 999 samples (15.8%)

#### Hip-Hop (2,297 samples)
- FMA Medium: 1,197 samples (52.1%)
- FMA Small: 1,000 samples (43.5%)
- GTZAN: 100 samples (4.4%)

#### Experimental (2,250 samples)
- FMA Medium: 1,251 samples (55.6%)
- FMA Small: 999 samples (44.4%)

#### Folk (1,518 samples)
- FMA Small: 1,000 samples (65.9%)
- FMA Medium: 518 samples (34.1%)

#### Pop (1,386 samples)
- FMA Small: 1,000 samples (72.2%)
- FMA Medium: 186 samples (13.4%)
- Indian (Bollypop): 100 samples (7.2%)
- GTZAN: 100 samples (7.2%)

#### Instrumental (1,349 samples)
- FMA Small: 1,000 samples (74.1%)
- FMA Medium: 349 samples (25.9%)

#### World (1,218 samples)
- FMA Small (International): 1,000 samples (82.1%)
- Indian (Ghazal): 100 samples (8.2%)
- Indian (Sufi): 100 samples (8.2%)
- FMA Medium (International): 18 samples (1.5%)

#### Classical (919 samples)
- FMA Medium: 619 samples (67.4%)
- Indian (Carnatic): 100 samples (10.9%)
- Indian (Semiclassical): 100 samples (10.9%)
- GTZAN: 100 samples (10.9%)

---

## Dataset Characteristics

### Strengths

✅ **Large Scale:** 26,484 samples provide substantial data for training
✅ **Multi-Source:** Combines diverse music collections with different characteristics
✅ **Real-World Distribution:** Reflects actual music consumption patterns (imbalanced)
✅ **Cultural Diversity:** Includes Western and Indian music traditions
✅ **Rich Features:** 45 PCA components capture comprehensive audio characteristics
✅ **Traceable Sources:** Each sample tagged with source dataset for analysis

### Challenges

⚠️ **Highly Imbalanced:** Rock and Electronic dominate (51% combined)
⚠️ **Feature Padding:** Smaller datasets have zero-padded features
⚠️ **Label Abstraction:** Some cultural nuances lost in genre unification
⚠️ **Rare Classes:** 8 genres have <1% representation each
⚠️ **Dataset Bias:** Majority samples from FMA Medium (64.1%)

### Best Practices for Use

1. **Class Imbalance Handling:**
   - Use stratified sampling for train/test splits
   - Consider class weights in loss functions
   - Evaluate using balanced metrics (F1, balanced accuracy)
   - Try oversampling (SMOTE) or undersampling techniques

2. **Validation Strategy:**
   - Use source-stratified k-fold cross-validation
   - Test generalization across different source datasets
   - Monitor per-genre performance, not just overall accuracy

3. **Clustering Analysis:**
   - Be aware that unsupervised methods will favor larger classes
   - Consider Silhouette score over purity for imbalanced data
   - Use k values exploring both coarse (5-10) and fine (15-20) granularities

4. **Feature Engineering:**
   - Consider masking zero-padded features for fair comparison
   - Analyze whether padding affects downstream tasks
   - May want to standardize features after combination

---

## Usage Examples

### Basic Loading

```python
import pandas as pd

# Load features
features_df = pd.read_csv('data/combined-set.csv')
print(f"Features shape: {features_df.shape}")

# Load labels
labels_df = pd.read_csv('data/combined-labels.csv')
print(f"Labels shape: {labels_df.shape}")

# Quick stats
print(labels_df['label'].value_counts())
print(labels_df['source_dataset'].value_counts())
```

### Stratified Split

```python
from sklearn.model_selection import train_test_split

X = features_df.values
y = labels_df['label'].values

# Stratified split maintaining genre proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### Clustering with Combined Dataset

```python
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Prepare data
X = features_df.values
y_true = labels_df['label'].values

# Cluster with k=19 (matching number of genres)
kmeans = KMeans(n_clusters=19, random_state=42)
y_pred = kmeans.fit_predict(X)

# Evaluate
ari = adjusted_rand_score(y_true, y_pred)
print(f"Adjusted Rand Index: {ari:.3f}")
```

### Source-Specific Analysis

```python
# Analyze performance by source dataset
for source in labels_df['source_dataset'].unique():
    mask = labels_df['source_dataset'] == source
    X_source = features_df[mask].values
    y_source = labels_df[mask]['label'].values
    
    print(f"\n{source}:")
    print(f"  Samples: {len(X_source)}")
    print(f"  Genres: {len(set(y_source))}")
```

---

## Comparison with Individual Datasets

### Advantages of Combined Dataset

| Aspect | Individual Datasets | Combined Dataset |
|--------|-------------------|------------------|
| Size | 500 - 16,988 samples | 26,484 samples |
| Diversity | Single source bias | Multi-source diversity |
| Genres | 5 - 16 genres | 19 unified genres |
| Cultural Coverage | Limited | Western + Indian |
| Real-World Relevance | Varies | High (imbalanced) |
| Generalization | Dataset-specific | Cross-dataset |

### When to Use Combined vs. Individual

**Use Combined Dataset When:**
- You need maximum data for training
- Testing cross-dataset generalization
- Exploring diverse music styles
- Real-world imbalanced scenarios
- Large-scale clustering experiments

**Use Individual Datasets When:**
- You need perfectly balanced data (Indian, FMA Small, GTZAN)
- Focusing on specific music traditions (Indian)
- Benchmark comparisons (GTZAN)
- Controlled experimental conditions
- Dataset-specific research questions

---

## Technical Specifications

### File Formats

#### combined-set.csv
```
PC1,PC2,PC3,PC4,...,PC45
1.387,-3.468,1.277,2.519,...,0.0
-2.536,0.454,2.675,1.683,...,0.0
...
```
- **Format:** CSV with header
- **Rows:** 26,484 (samples)
- **Columns:** 45 (PCA components)
- **Encoding:** UTF-8
- **Missing Values:** None

#### combined-labels.csv
```
label,source_dataset
Pop,indian
Classical,indian
World,indian
...
```
- **Format:** CSV with header
- **Rows:** 26,484 (samples)
- **Columns:** 2 (label, source_dataset)
- **Encoding:** UTF-8
- **Missing Values:** None

### Data Quality

✅ **Validated:** All rows have correct dimensions
✅ **Aligned:** Labels match features row-by-row
✅ **Clean:** No missing or NaN values
✅ **Standardized:** Consistent naming conventions
✅ **Traceable:** Source dataset preserved for each sample

---

## Research Applications

### Recommended Research Questions

1. **Cross-Dataset Generalization:** How well do models trained on one source generalize to others?
2. **Imbalanced Learning:** What techniques work best for highly imbalanced music genre classification?
3. **Genre Relationships:** Which genres cluster together naturally across datasets?
4. **Cultural Distinctions:** Can models distinguish Western vs. Indian classical music?
5. **Transfer Learning:** Does pre-training on FMA help with Indian music classification?

### Clustering Experiment Ideas

1. **Multi-Granularity Clustering:**
   - Coarse: k=5-10 (major genre families)
   - Medium: k=15-20 (standard genres)
   - Fine: k=25-30 (sub-genres)

2. **Hierarchical Analysis:**
   - First level: Major categories (Rock/Electronic/World/Classical)
   - Second level: Specific genres within categories

3. **Source-Stratified Evaluation:**
   - Cluster all data together
   - Evaluate clustering quality per source dataset
   - Compare consistency across sources

---

## Limitations and Considerations

### Known Limitations

1. **Feature Padding:** Zero-padding may introduce artifacts in distance calculations
2. **Label Subjectivity:** Genre boundaries are inherently fuzzy
3. **Cultural Bias:** More Western music than other traditions
4. **Temporal Bias:** Different datasets from different time periods
5. **Class Imbalance:** May require special handling for rare genres

### Ethical Considerations

- **Representation:** Dataset has limited non-Western music representation
- **Genre Stereotyping:** Unified labels may oversimplify cultural music traditions
- **Bias Amplification:** Model may perform poorly on underrepresented genres

---

## Version History

### Version 1.0 (December 5, 2025)
- Initial release
- Combined 4 datasets: Indian, FMA Small, FMA Medium, GTZAN
- Unified 31 original labels into 19 standardized genres
- Total 26,484 samples with 45 features each

---

## References

1. **FMA Dataset:** Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). FMA: A dataset for music analysis. ISMIR.

2. **GTZAN Dataset:** Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. IEEE Transactions on speech and audio processing.

3. **Indian Music Dataset:** Custom curated collection of traditional and contemporary Indian music.

---

## Citation

If you use this combined dataset in your research, please cite all original datasets:

```bibtex
@inproceedings{fma_dataset,
  title={FMA: A Dataset for Music Analysis},
  author={Defferrard, Micha{\"e}l and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
  booktitle={18th International Society for Music Information Retrieval Conference (ISMIR)},
  year={2017}
}

@article{gtzan_dataset,
  title={Musical genre classification of audio signals},
  author={Tzanetakis, George and Cook, Perry},
  journal={IEEE Transactions on speech and audio processing},
  volume={10},
  number={5},
  pages={293--302},
  year={2002}
}
```

---

## Contact & Support

For questions about the combined dataset:
- Check individual dataset documentation in `DATASET_LABELS_README.md`
- Review clustering notebook configuration in `NOTEBOOKS/step4-clustering-Experiments.ipynb`
- Refer to main project documentation in `README.md`

---

## Last Updated
December 5, 2025
