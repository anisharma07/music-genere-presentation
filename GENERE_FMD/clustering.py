"""
Clustering Module
Implements K-Means, Spectral Clustering, DBSCAN, and GMM algorithms.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class ClusteringEngine:
    """Comprehensive clustering engine with multiple algorithms."""

    def __init__(self, data_path='cleaned_features.csv', n_clusters=10, random_state=42):
        """
        Initialize the clustering engine.

        Parameters:
        -----------
        data_path : str
            Path to the cleaned features CSV file
        n_clusters : int
            Number of clusters for algorithms that require it
        random_state : int
            Random state for reproducibility
        """
        self.data_path = data_path
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.df = None
        self.X = None
        self.X_pca = None
        self.scaler = StandardScaler()
        self.pca = None
        self.track_ids = None

    def load_and_preprocess(self, n_components=20):
        """
        Load data and perform preprocessing (standardization + PCA).

        Parameters:
        -----------
        n_components : int
            Number of PCA components
        """
        print("\n" + "="*70)
        print("LOADING AND PREPROCESSING DATA")
        print("="*70)

        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded data shape: {self.df.shape}")

        # Separate track_ids if present
        if 'track_id' in self.df.columns:
            self.track_ids = self.df['track_id'].values
            self.X = self.df.drop('track_id', axis=1).values
        else:
            self.track_ids = np.arange(len(self.df))
            self.X = self.df.values

        print(f"Feature matrix shape: {self.X.shape}")

        # Standardize features
        self.X = self.scaler.fit_transform(self.X)
        print("✓ Features standardized (mean=0, std=1)")

        # Apply PCA
        self.pca = PCA(n_components=n_components,
                       random_state=self.random_state)
        self.X_pca = self.pca.fit_transform(self.X)

        explained_var = self.pca.explained_variance_ratio_.sum()
        print(
            f"✓ PCA applied: {self.X.shape[1]} features → {n_components} components")
        print(f"  Explained variance: {explained_var*100:.2f}%")

        return self.X_pca

    def tune_dbscan_parameters(self, X, k=5, output_path='dbscan_k_distance.png'):
        """
        Use k-distance plot to determine optimal eps parameter for DBSCAN.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        k : int
            Number of nearest neighbors (typically min_samples)
        output_path : str
            Path to save the k-distance plot

        Returns:
        --------
        float
            Suggested eps value (elbow point)
        """
        print(f"\nTuning DBSCAN parameters using k-distance plot (k={k})...")

        # Calculate k-nearest neighbors
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)

        # Sort distances
        distances = np.sort(distances[:, k-1], axis=0)

        # Find elbow point (simplified approach: use 90th percentile)
        suggested_eps = np.percentile(distances, 90)

        # Plot k-distance graph
        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.axhline(y=suggested_eps, color='r', linestyle='--',
                    label=f'Suggested eps={suggested_eps:.3f} (90th percentile)')
        plt.ylabel(f'{k}-NN Distance')
        plt.xlabel('Data Points sorted by distance')
        plt.title(f'K-distance Graph for DBSCAN Parameter Tuning (k={k})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"K-distance plot saved to: {output_path}")
        plt.close()

        print(f"Suggested eps parameter: {suggested_eps:.3f}")
        print(
            f"Consider trying eps in range: [{suggested_eps*0.5:.3f}, {suggested_eps*1.5:.3f}]")

        return suggested_eps

    def split_data(self, test_size=0.2, return_indices=False):
        """
        Split data into train and test sets.

        Parameters:
        -----------
        test_size : float
            Proportion of data for testing
        return_indices : bool
            Whether to return indices

        Returns:
        --------
        tuple
            (X_train, X_test) or (X_train, X_test, train_idx, test_idx)
        """
        if return_indices:
            indices = np.arange(len(self.X_pca))
            X_train, X_test, idx_train, idx_test = train_test_split(
                self.X_pca, indices, test_size=test_size,
                random_state=self.random_state
            )
            return X_train, X_test, idx_train, idx_test
        else:
            X_train, X_test = train_test_split(
                self.X_pca, test_size=test_size,
                random_state=self.random_state
            )
            return X_train, X_test

    def kmeans_clustering(self, X, variant='standard'):
        """
        Perform K-Means clustering.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        variant : str
            'standard' or 'minibatch'

        Returns:
        --------
        tuple
            (labels, model)
        """
        print(f"\nRunning K-Means ({variant})...")

        if variant == 'minibatch':
            model = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                batch_size=256,
                max_iter=300
            )
        else:
            model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )

        labels = model.fit_predict(X)
        print(
            f"✓ Clustering complete. Found {len(np.unique(labels))} clusters")

        return labels, model

    def spectral_clustering(self, X):
        """
        Perform Spectral Clustering.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix

        Returns:
        --------
        tuple
            (labels, model)
        """
        print(f"\nRunning Spectral Clustering...")

        model = SpectralClustering(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            affinity='nearest_neighbors',
            n_neighbors=10,
            assign_labels='kmeans'
        )

        labels = model.fit_predict(X)
        print(
            f"✓ Clustering complete. Found {len(np.unique(labels))} clusters")

        return labels, model

    def dbscan_clustering(self, X, eps=0.5, min_samples=5):
        """
        Perform DBSCAN clustering.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        eps : float
            Maximum distance between samples
        min_samples : int
            Minimum samples in a neighborhood

        Returns:
        --------
        tuple
            (labels, model)
        """
        print(f"\nRunning DBSCAN (eps={eps}, min_samples={min_samples})...")

        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = model.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        print(f"✓ Clustering complete. Found {n_clusters} clusters")
        print(f"  Noise points: {n_noise} ({n_noise/len(labels)*100:.2f}%)")

        return labels, model

    def gmm_clustering(self, X):
        """
        Perform Gaussian Mixture Model clustering.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix

        Returns:
        --------
        tuple
            (labels, model)
        """
        print(f"\nRunning GMM...")

        model = GaussianMixture(
            n_components=self.n_clusters,
            random_state=self.random_state,
            covariance_type='full',
            max_iter=200
        )

        model.fit(X)
        labels = model.predict(X)

        print(
            f"✓ Clustering complete. Found {len(np.unique(labels))} clusters")
        print(f"  Converged: {model.converged_}")

        return labels, model

    def run_all_algorithms(self, X, dbscan_eps=0.5, dbscan_min_samples=5):
        """
        Run all clustering algorithms on the given data.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        dbscan_eps : float
            DBSCAN epsilon parameter
        dbscan_min_samples : int
            DBSCAN min_samples parameter

        Returns:
        --------
        dict
            Dictionary containing results for all algorithms
        """
        print("\n" + "="*70)
        print("RUNNING ALL CLUSTERING ALGORITHMS")
        print("="*70)

        results = {}

        # K-Means (standard)
        labels_km, model_km = self.kmeans_clustering(X, variant='standard')
        results['K-Means'] = {
            'labels': labels_km,
            'model': model_km,
            'n_clusters': self.n_clusters
        }

        # MiniBatch K-Means
        labels_mbkm, model_mbkm = self.kmeans_clustering(
            X, variant='minibatch')
        results['MiniBatch K-Means'] = {
            'labels': labels_mbkm,
            'model': model_mbkm,
            'n_clusters': self.n_clusters
        }

        # Spectral Clustering
        labels_sc, model_sc = self.spectral_clustering(X)
        results['Spectral Clustering'] = {
            'labels': labels_sc,
            'model': model_sc,
            'n_clusters': self.n_clusters
        }

        # DBSCAN
        labels_db, model_db = self.dbscan_clustering(X, eps=dbscan_eps,
                                                     min_samples=dbscan_min_samples)
        n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
        results['DBSCAN'] = {
            'labels': labels_db,
            'model': model_db,
            'n_clusters': n_clusters_db
        }

        # GMM
        labels_gmm, model_gmm = self.gmm_clustering(X)
        results['GMM'] = {
            'labels': labels_gmm,
            'model': model_gmm,
            'n_clusters': self.n_clusters
        }

        print("\n✓ All algorithms completed successfully!")

        return results

    def experiment_with_splits(self, splits=[0.5, 0.4, 0.3, 0.2],
                               dbscan_eps=0.5, dbscan_min_samples=5):
        """
        Run experiments with different train-test splits.

        Parameters:
        -----------
        splits : list
            List of test split proportions (e.g., [0.5, 0.4, 0.3, 0.2])
        dbscan_eps : float
            DBSCAN epsilon parameter
        dbscan_min_samples : int
            DBSCAN min_samples parameter

        Returns:
        --------
        dict
            Dictionary containing results for each split
        """
        print("\n" + "="*70)
        print("EXPERIMENTAL SETUP: DIFFERENT TRAIN-TEST SPLITS")
        print("="*70)

        all_results = {}

        for test_size in splits:
            train_size = 1 - test_size
            split_name = f"{int(train_size*100)}-{int(test_size*100)}"

            print(f"\n{'='*70}")
            print(
                f"SPLIT: {split_name} (Train: {int(train_size*100)}%, Test: {int(test_size*100)}%)")
            print(f"{'='*70}")

            # Split data and get indices so we can map back to original track ids
            X_train, X_test, idx_train, idx_test = self.split_data(
                test_size=test_size, return_indices=True)
            print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

            # Run all algorithms on training data
            results = self.run_all_algorithms(X_train,
                                              dbscan_eps=dbscan_eps,
                                              dbscan_min_samples=dbscan_min_samples)

            all_results[split_name] = {
                'X_train': X_train,
                'X_test': X_test,
                'train_idx': idx_train,
                'test_idx': idx_test,
                'results': results
            }

        return all_results

    def save_results(self, results, output_path='clustering_results.csv'):
        """
        Save clustering results to CSV.

        Parameters:
        -----------
        results : dict
            Dictionary containing clustering results
        output_path : str
            Path to save the results
        """
        # Prepare data for saving
        data_to_save = []

        for algo_name, result in results.items():
            labels = result['labels']

            for i, label in enumerate(labels):
                data_to_save.append({
                    'algorithm': algo_name,
                    'sample_index': i,
                    'cluster_label': label
                })

        df_results = pd.DataFrame(data_to_save)
        df_results.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    engine = ClusteringEngine(data_path='cleaned_features.csv', n_clusters=10)

    # Load and preprocess data
    X_pca = engine.load_and_preprocess(n_components=20)

    # Run experiments with different splits
    all_results = engine.experiment_with_splits(
        splits=[0.5, 0.4, 0.3, 0.2],
        dbscan_eps=0.5,
        dbscan_min_samples=5
    )

    print("\n" + "="*70)
    print("CLUSTERING EXPERIMENTS COMPLETE")
    print("="*70)
