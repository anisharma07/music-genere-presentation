# Step 3: Handle Missing & Incorrect Data + Outlier Detection Results

## Overview
This step focused on data quality assurance, handling missing/incorrect data, and detecting and removing outliers using statistical methods.

## Input Data
All datasets loaded from `/data/` folder (cleaned in Step 2):
- `gtzan_clean.csv`
- `fma_small_clean.csv`
- `fma_medium_clean.csv`
- `instrumental_clean.csv`

## Data Quality Assessment

### Initial Quality Check

| Dataset      | Missing (NaN) | Infinity (±∞) | Empty Rows | Duplicates |
|--------------|---------------|---------------|------------|------------|
| GTZAN        | 0             | 0             | 0          | 13         |
| FMA Small    | 0             | 0             | 0          | 12         |
| FMA Medium   | 0             | 0             | 0          | 73         |
| Instrumental | 0             | 0             | 0          | 6          |

**Finding:** All datasets were already clean from Step 2, but contained some duplicate rows.

### Cleaning Operations Applied

1. **Replace ±∞ with NaN** - None found
2. **Remove empty rows** - None found
3. **Remove rows with NaN** - None found
4. **Remove duplicate rows** - Removed 104 total duplicates

**After Cleaning:**
- GTZAN: 999 → 986 rows (13 duplicates removed)
- FMA Small: 7,997 → 7,985 rows (12 duplicates removed)
- FMA Medium: 24,985 → 24,912 rows (73 duplicates removed)
- Instrumental: 502 → 496 rows (6 duplicates removed)

## Outlier Detection Analysis

### Method 1: Z-Score Method
**Threshold:** Z-score > 3 (99.7% confidence interval)

| Dataset      | Total Rows | Outliers | Percentage |
|--------------|------------|----------|------------|
| GTZAN        | 986        | 165      | 16.73%     |
| FMA Small    | 7,985      | 1,801    | 22.55%     |
| FMA Medium   | 24,912     | 5,813    | 23.33%     |
| Instrumental | 496        | 96       | 19.35%     |

**Top Features with Outliers (Z-Score):**
- **GTZAN:** mfcc13_std (17), mfcc14_std (16), mfcc17_std (15)
- **FMA Small:** mfcc19_std (118), mfcc20_std (118), mfcc18_std (117)
- **FMA Medium:** mfcc6_mean (459), mfcc3_mean (403), mfcc20_std (378)
- **Instrumental:** mfcc17_mean (10), mfcc8_std (9), mfcc6_mean (9)

### Method 2: IQR (Interquartile Range) Method
**Threshold:** Values < Q1 - 1.5×IQR or > Q3 + 1.5×IQR

| Dataset      | Total Rows | Outliers | Percentage |
|--------------|------------|----------|------------|
| GTZAN        | 986        | 298      | 30.22%     |
| FMA Small    | 7,985      | 3,267    | 40.91%     |
| FMA Medium   | 24,912     | 10,364   | 41.60%     |
| Instrumental | 496        | 192      | 38.71%     |

**Top Features with Outliers (IQR):**
- **GTZAN:** mfcc18_std (33), mfcc13_std (31), chroma11_std (31)
- **FMA Small:** mfcc4_mean (289), mfcc20_std (284), mfcc19_std (267)
- **FMA Medium:** mfcc4_mean (1,139), mfcc6_mean (1,115), mfcc20_std (959)
- **Instrumental:** mfcc8_std (29), mfcc7_std (23), mfcc9_std (23)

### Comparison: Z-Score vs IQR

| Dataset      | Total Rows | Z-Score Outliers | Z-Score % | IQR Outliers | IQR % |
|--------------|------------|------------------|-----------|--------------|-------|
| GTZAN        | 986        | 165              | 16.73%    | 298          | 30.22% |
| FMA Small    | 7,985      | 1,801            | 22.55%    | 3,267        | 40.91% |
| FMA Medium   | 24,912     | 5,813            | 23.33%    | 10,364       | 41.60% |
| Instrumental | 496        | 96               | 19.35%    | 192          | 38.71% |

**Key Insight:** IQR method consistently identifies ~2x more outliers than Z-score method. IQR is more robust for non-normally distributed data, which is common in audio features.

## Outlier Removal

**Method Used:** IQR Method (more robust for audio features)

### Results After Outlier Removal

| Dataset      | Before | After | Removed | Removal % |
|--------------|--------|-------|---------|-----------|
| GTZAN        | 986    | 688   | 298     | 30.22%    |
| FMA Small    | 7,985  | 4,718 | 3,267   | 40.91%    |
| FMA Medium   | 24,912 | 14,548| 10,364  | 41.60%    |
| Instrumental | 496    | 304   | 192     | 38.71%    |

## Final Dataset Statistics

### GTZAN
- **Final Size:** 688 rows × 73 columns
- **Unique Labels:** 10 genres
- **Label Distribution:**
  - Disco: 87
  - Country: 79
  - Rock: 78
  - Pop: 75
  - Hip-hop: 73
  - Reggae: 72
  - Metal: 71
  - Jazz: 70
  - Blues: 64
  - Classical: 19 ⚠️ (significant reduction)

### FMA Small
- **Final Size:** 4,718 rows × 74 columns
- **Labels:** All "unknown" (metadata not available)

### FMA Medium
- **Final Size:** 14,548 rows × 74 columns
- **Labels:** All "unknown" (metadata not available)

### Instrumental
- **Final Size:** 304 rows × 74 columns
- **Labels:** All "instrumental"

## Overall Processing Summary

| Dataset      | Original | After Step 2 | After Step 3 | Total Removed | Total % Removed |
|--------------|----------|--------------|--------------|---------------|-----------------|
| GTZAN        | 999      | 999          | 688          | 311           | 31.13%          |
| FMA Small    | 7,997    | 7,997        | 4,718        | 3,279         | 41.00%          |
| FMA Medium   | 24,985   | 24,985       | 14,548       | 10,437        | 41.77%          |
| Instrumental | 502      | 502          | 304          | 198           | 39.44%          |
| **Total**    | **34,483** | **34,483** | **20,258**   | **14,225**    | **41.25%**      |

## Output Files

All processed datasets saved to: `/results/step3/`

1. **gtzan_no_missing.csv** - 688 rows × 73 columns
2. **fma_small_no_missing.csv** - 4,718 rows × 74 columns
3. **fma_medium_no_missing.csv** - 14,548 rows × 74 columns
4. **instrumental_no_missing.csv** - 304 rows × 74 columns

## Key Findings

### Strengths
- ✓ **No missing values** in any dataset
- ✓ **No infinity values** in any dataset
- ✓ **Removed 104 duplicate rows** across all datasets
- ✓ **Comprehensive outlier detection** using two methods
- ✓ **Robust outlier removal** using IQR method
- ✓ All 10 genres still represented in GTZAN

### Concerns
- ⚠️ **Classical genre severely reduced** in GTZAN (100 → 19 samples)
  - May indicate genuine outliers or unique characteristics of classical music
  - Could impact model performance for this genre
- ⚠️ **High outlier percentage** (38-42%) in FMA and Instrumental datasets
  - Suggests high variability in audio features
  - IQR method may be aggressive for these datasets

### Observations
- **MFCC features** (especially high coefficients) had most outliers
- **Standard deviation features** more prone to outliers than means
- **Chroma features** also showed significant outliers
- Z-score detected 17-23% outliers vs IQR's 30-42%

## Recommendations

### For Modeling
1. **GTZAN dataset:**
   - Use stratified sampling to handle imbalanced classes
   - Consider class weights for Classical genre (only 19 samples)
   - May need separate evaluation for Classical
   
2. **FMA datasets:**
   - Good for unsupervised learning (clustering, dimensionality reduction)
   - Large enough for deep learning after outlier removal
   
3. **Instrumental dataset:**
   - Suitable for binary classification tasks
   - Reduced to 304 samples but still usable

### Alternative Approaches
1. **Consider Z-score method** if IQR is too aggressive
2. **Use ensemble outlier detection** (combine multiple methods)
3. **Apply outlier removal per genre** to preserve class balance
4. **Keep separate versions** with and without outlier removal for comparison

## Visualizations Generated

Box plots created for GTZAN dataset showing outlier distribution for:
- `tempo` - 12 outliers
- `rms_mean` - 5 outliers
- `spec_centroid_mean` - 1 outlier
- `zcr_mean` - 6 outliers

## Next Steps

1. **Exploratory Data Analysis (EDA)** - Detailed feature analysis and visualization
2. **Feature Selection** - Identify most important features for classification
3. **Dimensionality Reduction** - Apply PCA/t-SNE for visualization
4. **Feature Scaling** - Normalize/standardize features for modeling
5. **Model Training** - Build classification models on GTZAN
6. **Clustering Analysis** - Apply unsupervised learning to FMA datasets

---

**Status:** ✓ All datasets cleaned, duplicates removed, outliers detected and removed
**Generated:** November 29, 2025
**Data Quality:** Excellent - Ready for modeling
