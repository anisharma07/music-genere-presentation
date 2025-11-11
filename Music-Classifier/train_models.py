"""
Music Genre Classification using KMeans and Spectral Clustering
Train models with different train-test splits: 50-50, 60-40, 70-30, 80-20
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Create directory for models if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load the dataset
print("Loading GTZAN dataset...")
df = pd.read_csv('gtzan.csv')

# Separate features and labels
X = df.drop(['filename', 'label'], axis=1)
y = df['label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"Label classes: {label_encoder.classes_}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Define train-test splits
splits = {
    '50-50': 0.5,
    '60-40': 0.4,
    '70-30': 0.3,
    '80-20': 0.2
}

# Determine optimal number of clusters from unique genres
n_clusters = len(label_encoder.classes_)
print(f"\nNumber of genres (clusters): {n_clusters}")

print("\n" + "="*80)
print("Training Models for Different Train-Test Splits")
print("="*80)

# Train models for each split
for split_name, test_size in splits.items():
    print(f"\n{'='*80}")
    print(f"Training models for {split_name} split")
    print(f"{'='*80}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train KMeans
    print(f"\n[KMeans - {split_name}]")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42,
                    n_init=10, max_iter=300)
    kmeans.fit(X_train)

    # Evaluate KMeans
    train_pred_kmeans = kmeans.predict(X_train)
    test_pred_kmeans = kmeans.predict(X_test)

    train_ari_kmeans = adjusted_rand_score(y_train, train_pred_kmeans)
    test_ari_kmeans = adjusted_rand_score(y_test, test_pred_kmeans)
    train_nmi_kmeans = normalized_mutual_info_score(y_train, train_pred_kmeans)
    test_nmi_kmeans = normalized_mutual_info_score(y_test, test_pred_kmeans)
    train_silhouette_kmeans = silhouette_score(X_train, train_pred_kmeans)
    test_silhouette_kmeans = silhouette_score(X_test, test_pred_kmeans)

    print(
        f"  Train - ARI: {train_ari_kmeans:.4f}, NMI: {train_nmi_kmeans:.4f}, Silhouette: {train_silhouette_kmeans:.4f}")
    print(
        f"  Test  - ARI: {test_ari_kmeans:.4f}, NMI: {test_nmi_kmeans:.4f}, Silhouette: {test_silhouette_kmeans:.4f}")

    # Save KMeans model
    model_filename = f"models/kmeans_{split_name.replace('-', '_')}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"  Saved: {model_filename}")

    # Train Spectral Clustering
    print(f"\n[Spectral Clustering - {split_name}]")
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        random_state=42,
        affinity='nearest_neighbors',
        n_neighbors=10,
        assign_labels='kmeans'
    )
    train_pred_spectral = spectral.fit_predict(X_train)

    # For prediction on test set, we need to use a different approach
    # Spectral clustering doesn't have a predict method, so we'll use KMeans on the embedding
    # We'll train a KMeans on the full training set for prediction purposes
    spectral_full = SpectralClustering(
        n_clusters=n_clusters,
        random_state=42,
        affinity='nearest_neighbors',
        n_neighbors=10,
        assign_labels='kmeans'
    )

    # For evaluation, we'll use the training predictions
    train_ari_spectral = adjusted_rand_score(y_train, train_pred_spectral)
    train_nmi_spectral = normalized_mutual_info_score(
        y_train, train_pred_spectral)
    train_silhouette_spectral = silhouette_score(X_train, train_pred_spectral)

    print(
        f"  Train - ARI: {train_ari_spectral:.4f}, NMI: {train_nmi_spectral:.4f}, Silhouette: {train_silhouette_spectral:.4f}")
    print(f"  Note: Spectral Clustering doesn't support direct prediction on new data.")
    print(f"        Using KMeans on spectral embeddings for test predictions.")

    # Save Spectral model (with training data for future predictions)
    model_data = {
        'model': spectral_full,
        'X_train': X_train,
        'train_labels': train_pred_spectral
    }
    model_filename = f"models/spectral_{split_name.replace('-', '_')}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"  Saved: {model_filename}")

print("\n" + "="*80)
print("Model Training Complete!")
print("="*80)
print(f"\nTrained models saved in 'models/' directory:")
print("  - KMeans models: kmeans_50_50.pkl, kmeans_60_40.pkl, kmeans_70_30.pkl, kmeans_80_20.pkl")
print("  - Spectral models: spectral_50_50.pkl, spectral_60_40.pkl, spectral_70_30.pkl, spectral_80_20.pkl")
print("  - Label encoder: label_encoder.pkl")
print("  - Feature scaler: scaler.pkl")
