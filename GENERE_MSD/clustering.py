"""
Clustering algorithms implementation
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any
import logging
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MusicGenreClusterer:
    """Clustering algorithms for music genre discovery"""

    def __init__(self, config: Dict, random_state: int = 42):
        self.config = config
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.models = {}

    def preprocess_features(self, X: np.ndarray, n_components: int = 20,
                            fit: bool = True) -> np.ndarray:
        """Standardize and apply PCA to features"""

        logger.info(f"Preprocessing features... Original shape: {X.shape}")

        # Standardization
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        # PCA
        if fit:
            self.pca = PCA(n_components=n_components,
                           random_state=self.random_state)
            X_reduced = self.pca.fit_transform(X_scaled)

            explained_var = np.sum(self.pca.explained_variance_ratio_)
            logger.info(
                f"PCA: {n_components} components explain {explained_var:.2%} of variance")
        else:
            X_reduced = self.pca.transform(X_scaled)

        logger.info(f"After PCA: {X_reduced.shape}")

        return X_reduced

    def split_data(self, X: np.ndarray, train_ratio: float = 0.7) -> Tuple:
        """Split data into train and test sets"""

        X_train, X_test = train_test_split(
            X, train_size=train_ratio, random_state=self.random_state
        )

        logger.info(f"Data split: Train {X_train.shape}, Test {X_test.shape}")

        return X_train, X_test

    def kmeans_clustering(self, X: np.ndarray, n_clusters: int = 10) -> Tuple:
        """K-Means clustering"""

        logger.info(f"Running K-Means with {n_clusters} clusters...")

        model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            max_iter=self.config['kmeans'].get('max_iter', 300),
            n_init=self.config['kmeans'].get('n_init', 10)
        )

        labels = model.fit_predict(X)

        self.models['kmeans'] = model

        logger.info(
            f"K-Means complete. Unique labels: {len(np.unique(labels))}")

        return labels, model

    def minibatch_kmeans_clustering(self, X: np.ndarray, n_clusters: int = 10) -> Tuple:
        """MiniBatch K-Means clustering (faster for large datasets)"""

        logger.info(f"Running MiniBatch K-Means with {n_clusters} clusters...")

        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            max_iter=self.config['minibatch_kmeans'].get('max_iter', 300),
            batch_size=self.config['minibatch_kmeans'].get('batch_size', 100)
        )

        labels = model.fit_predict(X)

        self.models['minibatch_kmeans'] = model

        logger.info(
            f"MiniBatch K-Means complete. Unique labels: {len(np.unique(labels))}")

        return labels, model

    def spectral_clustering(self, X: np.ndarray, n_clusters: int = 10) -> Tuple:
        """Spectral clustering"""

        logger.info(
            f"Running Spectral Clustering with {n_clusters} clusters...")

        # Limit samples for spectral clustering (computationally expensive)
        if len(X) > 5000:
            logger.warning(
                f"Spectral clustering on {len(X)} samples may be slow. Consider subsampling.")

        model = SpectralClustering(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=self.config['spectral'].get('n_init', 10),
            affinity=self.config['spectral'].get(
                'affinity', 'nearest_neighbors'),
            assign_labels='kmeans'
        )

        labels = model.fit_predict(X)

        self.models['spectral'] = model

        logger.info(
            f"Spectral Clustering complete. Unique labels: {len(np.unique(labels))}")

        return labels, model

    def dbscan_clustering(self, X: np.ndarray, eps: float = 0.5,
                          min_samples: int = 5) -> Tuple:
        """DBSCAN clustering"""

        logger.info(
            f"Running DBSCAN with eps={eps}, min_samples={min_samples}...")

        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=self.config['dbscan'].get('metric', 'euclidean'),
            n_jobs=-1
        )

        labels = model.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        self.models['dbscan'] = model

        logger.info(
            f"DBSCAN complete. Clusters: {n_clusters}, Noise points: {n_noise}")

        return labels, model

    def gmm_clustering(self, X: np.ndarray, n_components: int = 10) -> Tuple:
        """Gaussian Mixture Model clustering"""

        logger.info(f"Running GMM with {n_components} components...")

        model = GaussianMixture(
            n_components=n_components,
            random_state=self.random_state,
            covariance_type=self.config['gmm'].get('covariance_type', 'full'),
            max_iter=self.config['gmm'].get('max_iter', 100)
        )

        model.fit(X)
        labels = model.predict(X)

        self.models['gmm'] = model

        logger.info(f"GMM complete. Unique labels: {len(np.unique(labels))}")

        return labels, model

    def run_all_algorithms(self, X: np.ndarray, n_clusters: int = 10) -> Dict:
        """Run all clustering algorithms"""

        results = {}

        # K-Means
        try:
            labels, model = self.kmeans_clustering(X, n_clusters)
            results['kmeans'] = {'labels': labels, 'model': model}
        except Exception as e:
            logger.error(f"K-Means failed: {str(e)}")
            results['kmeans'] = None

        # MiniBatch K-Means
        try:
            labels, model = self.minibatch_kmeans_clustering(X, n_clusters)
            results['minibatch_kmeans'] = {'labels': labels, 'model': model}
        except Exception as e:
            logger.error(f"MiniBatch K-Means failed: {str(e)}")
            results['minibatch_kmeans'] = None

        # Spectral Clustering (limit to smaller subset if too large)
        try:
            X_subset = X if len(X) <= 5000 else X[np.random.choice(
                len(X), 5000, replace=False)]
            labels_subset, model = self.spectral_clustering(
                X_subset, n_clusters)

            # For full dataset, use the model (though spectral doesn't predict directly)
            if len(X) > 5000:
                logger.warning(
                    "Spectral clustering only on subset. Full dataset labels not computed.")
                labels = np.full(len(X), -1)
                labels[:len(labels_subset)] = labels_subset
            else:
                labels = labels_subset

            results['spectral'] = {'labels': labels, 'model': model}
        except Exception as e:
            logger.error(f"Spectral Clustering failed: {str(e)}")
            results['spectral'] = None

        # DBSCAN
        try:
            # Use default parameters or first from config
            eps = self.config['dbscan']['eps'][0] if isinstance(
                self.config['dbscan']['eps'], list) else self.config['dbscan']['eps']
            min_samples = self.config['dbscan']['min_samples'][0] if isinstance(
                self.config['dbscan']['min_samples'], list) else self.config['dbscan']['min_samples']

            labels, model = self.dbscan_clustering(X, eps, min_samples)
            results['dbscan'] = {'labels': labels, 'model': model}
        except Exception as e:
            logger.error(f"DBSCAN failed: {str(e)}")
            results['dbscan'] = None

        # GMM
        try:
            labels, model = self.gmm_clustering(X, n_clusters)
            results['gmm'] = {'labels': labels, 'model': model}
        except Exception as e:
            logger.error(f"GMM failed: {str(e)}")
            results['gmm'] = None

        return results

    def save_models(self, output_dir: str):
        """Save trained models"""

        for name, model in self.models.items():
            model_path = os.path.join(output_dir, f'{name}_model.pkl')
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} model to {model_path}")

        # Save scaler and PCA
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        joblib.dump(self.pca, os.path.join(output_dir, 'pca.pkl'))

        logger.info("All models saved")

    def load_models(self, output_dir: str):
        """Load trained models"""

        # Load scaler and PCA
        self.scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))
        self.pca = joblib.load(os.path.join(output_dir, 'pca.pkl'))

        # Load clustering models
        model_files = {
            'kmeans': 'kmeans_model.pkl',
            'minibatch_kmeans': 'minibatch_kmeans_model.pkl',
            'spectral': 'spectral_model.pkl',
            'dbscan': 'dbscan_model.pkl',
            'gmm': 'gmm_model.pkl'
        }

        for name, filename in model_files.items():
            model_path = os.path.join(output_dir, filename)
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
                logger.info(f"Loaded {name} model")


def main():
    """Example usage"""
    from config import CLUSTERING_CONFIG, PCA_COMPONENTS, OUTPUT_DIR, MODELS_DIR

    # Load cleaned data
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'cleaned_features.csv'))

    # Get feature columns
    metadata_cols = ['track_id', 'artist_name', 'title']
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    X = df[feature_cols].values

    # Initialize clusterer
    clusterer = MusicGenreClusterer(CLUSTERING_CONFIG)

    # Preprocess
    X_processed = clusterer.preprocess_features(X, n_components=PCA_COMPONENTS)

    # Run all algorithms
    results = clusterer.run_all_algorithms(X_processed, n_clusters=10)

    # Save models
    clusterer.save_models(MODELS_DIR)

    # Save results
    for algo_name, result in results.items():
        if result:
            df[f'{algo_name}_cluster'] = result['labels']

    df.to_csv(os.path.join(OUTPUT_DIR, 'clustered_data.csv'), index=False)

    print("Clustering complete!")


if __name__ == "__main__":
    main()
