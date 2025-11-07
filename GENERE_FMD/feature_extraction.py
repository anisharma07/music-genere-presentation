"""
Feature Extraction Module for Music Genre Classification
Extracts audio features from MP3 files including MFCCs, Chroma, Tempo, and Spectral features.
"""

import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """Extract audio features from MP3 files."""

    def __init__(self, data_path='fma_small', sr=22050, duration=30):
        """
        Initialize the feature extractor.

        Parameters:
        -----------
        data_path : str
            Path to the FMA dataset directory
        sr : int
            Sample rate for audio loading
        duration : int
            Duration in seconds to load from each audio file
        """
        self.data_path = data_path
        self.sr = sr
        self.duration = duration

    def extract_features(self, audio_path):
        """
        Extract comprehensive audio features from a single audio file.

        Parameters:
        -----------
        audio_path : str
            Path to the audio file

        Returns:
        --------
        dict
            Dictionary containing extracted features
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sr,
                                 duration=self.duration)

            # Extract MFCCs (20 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)

            # Extract Delta MFCCs (temporal dynamics)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
            mfcc_delta_std = np.std(mfcc_delta, axis=1)

            # Extract Delta-Delta MFCCs (acceleration)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)
            mfcc_delta2_std = np.std(mfcc_delta2, axis=1)

            # Extract Chroma features (12 pitch classes)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)

            # Extract Spectral Centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_centroid_mean = np.mean(spectral_centroid)
            spectral_centroid_std = np.std(spectral_centroid)

            # Extract Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_rolloff_mean = np.mean(spectral_rolloff)
            spectral_rolloff_std = np.std(spectral_rolloff)

            # Extract Spectral Bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_bandwidth_mean = np.mean(spectral_bandwidth)
            spectral_bandwidth_std = np.std(spectral_bandwidth)

            # Extract Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zcr)
            zcr_std = np.std(zcr)

            # Extract Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            # Extract RMS Energy
            rms = librosa.feature.rms(y=y)
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)

            # Combine all features
            features = {}

            # MFCCs
            for i, (mean, std) in enumerate(zip(mfcc_mean, mfcc_std)):
                features[f'mfcc_{i}_mean'] = mean
                features[f'mfcc_{i}_std'] = std

            # Delta MFCCs
            for i, (mean, std) in enumerate(zip(mfcc_delta_mean, mfcc_delta_std)):
                features[f'mfcc_delta_{i}_mean'] = mean
                features[f'mfcc_delta_{i}_std'] = std

            # Delta-Delta MFCCs
            for i, (mean, std) in enumerate(zip(mfcc_delta2_mean, mfcc_delta2_std)):
                features[f'mfcc_delta2_{i}_mean'] = mean
                features[f'mfcc_delta2_{i}_std'] = std

            # Chroma
            for i, (mean, std) in enumerate(zip(chroma_mean, chroma_std)):
                features[f'chroma_{i}_mean'] = mean
                features[f'chroma_{i}_std'] = std

            # Spectral features
            features['spectral_centroid_mean'] = spectral_centroid_mean
            features['spectral_centroid_std'] = spectral_centroid_std
            features['spectral_rolloff_mean'] = spectral_rolloff_mean
            features['spectral_rolloff_std'] = spectral_rolloff_std
            features['spectral_bandwidth_mean'] = spectral_bandwidth_mean
            features['spectral_bandwidth_std'] = spectral_bandwidth_std

            # Other features
            features['zcr_mean'] = zcr_mean
            features['zcr_std'] = zcr_std
            features['tempo'] = tempo if isinstance(
                tempo, (int, float)) else tempo[0]
            features['rms_mean'] = rms_mean
            features['rms_std'] = rms_std

            return features

        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None

    def extract_all_features(self, max_files=None):
        """
        Extract features from all audio files in the dataset.

        Parameters:
        -----------
        max_files : int, optional
            Maximum number of files to process (for testing)

        Returns:
        --------
        pd.DataFrame
            DataFrame containing features for all audio files
        """
        features_list = []
        file_ids = []

        # Get all audio files
        audio_files = []
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.mp3'):
                    audio_files.append(os.path.join(root, file))

        # Limit files if specified
        if max_files:
            audio_files = audio_files[:max_files]

        print(f"Extracting features from {len(audio_files)} audio files...")

        # Extract features from each file
        for audio_path in tqdm(audio_files, desc="Processing audio files"):
            features = self.extract_features(audio_path)

            if features is not None:
                # Get file ID from path (e.g., 000002 from 000/000002.mp3)
                file_id = os.path.splitext(os.path.basename(audio_path))[0]

                features_list.append(features)
                file_ids.append(file_id)

        # Create DataFrame
        df = pd.DataFrame(features_list)
        df.insert(0, 'track_id', file_ids)

        print(f"Successfully extracted features from {len(df)} files")
        print(f"Total features per file: {len(df.columns) - 1}")

        return df

    def save_features(self, df, output_path='extracted_features.csv'):
        """
        Save extracted features to a CSV file.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing extracted features
        output_path : str
            Path to save the CSV file
        """
        df.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    extractor = AudioFeatureExtractor(data_path='fma_small')

    # Extract features from all files (or limit with max_files parameter)
    # For testing, you might want to use max_files=100
    features_df = extractor.extract_all_features(max_files=500)

    # Save features
    extractor.save_features(features_df, 'extracted_features.csv')

    # Display basic information
    print("\nFeature extraction complete!")
    print(f"Shape: {features_df.shape}")
    print(f"\nFirst few rows:")
    print(features_df.head())
    print(f"\nFeature statistics:")
    print(features_df.describe())
