# MY TOPIC IS: Unsupervised Music Genre Discovery Using Audio Feature Learning 

# DATA SET ANALYSIS:

Check for adequacy of data.
Check whether the dataset is imbalanced.
Generate all descriptive statistical analysis (which were discussed in the class)
Check for outliers in the data (use boxplot) Removing outliers (noisy data).
Removing null or irrelevant values in the columns. (Change null values to mean value of that column.)
If there is any missing data, either ignore the tuple or fill it with a mean value of the column.
Need for data augmentation.
Document  your results regarding data cleaning.
Identify the distribution pattern of data.
Find the sample mean X¯.
Find the 100p percentile for p = 0.75 and p= 0.25
Find the median M and the third quartile Q3.
Draw the box plot and identify the outliers.
Obtain the trimmed mean X¯ T . Decide on trimming fraction just enough to
eliminate the outliers and obtain the trimmed median M¯T .
obtain the trimmed standard deviation ST .
what can you say about the population from which the data arrived?
Conduct Correlation of important parameters. 

Document all your findings and recommendations. Include appropriate tables and graphs.

sample for reference
(https://www.kaggle.com/code/chrischapman12343/gender-inequality-data-cleaning-and-model-training#Data-Visuzalization

https://www.kaggle.com/code/priyanshusethi/gender-discrimination-a-case-study)

https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection?utm_medium=email&utm_source=gamma&utm_campaign=comp-recodai-2025

# IMPLEMENTATION:

    Experimental Setups:
 
       Conduct experiments : with different variants  of algorithms with training and testing : (50-50, 60-40, 70-30 and 80-20
distribution) and other options like randomization etc.  Cross validation
                    
 Use at least 6 evaluation metrics for each experiment.  Interpret your results Compare your results

# ALgorithms:
K-Means / MiniBatch K-Means
Spectral Clustering
DBSCAN (Density-Based Spatial Clustering)
Gaussian Mixture Model (GMM)

# METRICS TO BE MEASURED:

## Internal: 

Adjusted Rand Index (ARI)
Normalized Mutual Information (NMI)
Purity Index (PI)

## External: 

Silhouette Score
Davies–Bouldin Index (DBI)
Calinski–Harabasz Index (CHI)

# Workflow (for report)

Extract features (MFCCs, Chroma, Tempo, Spectral Centroid, etc.).

Reduce dimensions with PCA → 20D.

Apply all 4 clustering algorithms.

Evaluate using both internal and external metrics.

Compare results using tables and plots (Silhouette, DBI, ARI, etc.).

Example Output:
Algorithm	#Clusters (K)	Silhouette	Davies–Bouldin	Calinski–Harabasz	NMI	ARI	V-Measure	Cluster Accuracy
K-Means	10	0.41	0.86	240	0.52	0.47	0.55	0.60
Spectral Clustering	10	0.57	0.52	310	0.68	0.63	0.70	0.73
DBSCAN	auto	0.45	0.70	280	0.60	0.55	0.61	0.66
GMM	10	0.50	0.65	295	0.63	0.58	0.66	0.71