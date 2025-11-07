"""
Feature extraction from Million Song Dataset HDF5 files
"""

import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MillionSongFeatureExtractor:
    """Extract features from Million Song Dataset HDF5 files"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.h5_files = []

    def find_h5_files(self, max_files: int = None) -> List[str]:
        """Find all HDF5 files in the dataset directory"""
        logger.info(f"Searching for HDF5 files in {self.data_dir}")

        h5_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.h5'):
                    h5_files.append(os.path.join(root, file))
                    if max_files and len(h5_files) >= max_files:
                        break
            if max_files and len(h5_files) >= max_files:
                break

        self.h5_files = h5_files
        logger.info(f"Found {len(h5_files)} HDF5 files")
        return h5_files

    def extract_features_from_h5(self, h5_path: str) -> Dict:
        """
        Extract available features from a single HDF5 file

        Million Song Dataset contains:
        - Tempo, time signature, key, mode
        - Loudness, duration
        - Timbre features (12 coefficients)
        - Pitch features (12 coefficients)
        - Artist, song, album info
        """
        try:
            with h5py.File(h5_path, 'r') as h5:
                features = {}

                # Song metadata
                features['track_id'] = h5['analysis']['songs'][0]['track_id'].decode(
                    'utf-8')
                features['artist_name'] = h5['metadata']['songs'][0]['artist_name'].decode(
                    'utf-8')
                features['title'] = h5['metadata']['songs'][0]['title'].decode(
                    'utf-8')

                # Basic audio features
                features['duration'] = float(
                    h5['analysis']['songs'][0]['duration'])
                features['tempo'] = float(h5['analysis']['songs'][0]['tempo'])
                features['loudness'] = float(
                    h5['analysis']['songs'][0]['loudness'])
                features['key'] = int(h5['analysis']['songs'][0]['key'])
                features['mode'] = int(h5['analysis']['songs'][0]['mode'])
                features['time_signature'] = int(
                    h5['analysis']['songs'][0]['time_signature'])

                # Energy and danceability
                features['energy'] = float(
                    h5['analysis']['songs'][0]['energy'])
                features['danceability'] = float(
                    h5['analysis']['songs'][0]['danceability'])

                # Timbre features (segments timbre - 12D)
                segments_timbre = h5['analysis']['segments_timbre'][:]
                if len(segments_timbre) > 0:
                    # Statistical aggregations of timbre across segments
                    features['timbre_mean'] = np.mean(segments_timbre, axis=0)
                    features['timbre_std'] = np.std(segments_timbre, axis=0)
                    features['timbre_min'] = np.min(segments_timbre, axis=0)
                    features['timbre_max'] = np.max(segments_timbre, axis=0)
                    features['timbre_median'] = np.median(
                        segments_timbre, axis=0)
                else:
                    # Default values if no segments
                    features['timbre_mean'] = np.zeros(12)
                    features['timbre_std'] = np.zeros(12)
                    features['timbre_min'] = np.zeros(12)
                    features['timbre_max'] = np.zeros(12)
                    features['timbre_median'] = np.zeros(12)

                # Pitch features (segments pitches - 12D chroma-like)
                segments_pitches = h5['analysis']['segments_pitches'][:]
                if len(segments_pitches) > 0:
                    features['pitch_mean'] = np.mean(segments_pitches, axis=0)
                    features['pitch_std'] = np.std(segments_pitches, axis=0)
                    features['pitch_min'] = np.min(segments_pitches, axis=0)
                    features['pitch_max'] = np.max(segments_pitches, axis=0)
                    features['pitch_median'] = np.median(
                        segments_pitches, axis=0)
                else:
                    features['pitch_mean'] = np.zeros(12)
                    features['pitch_std'] = np.zeros(12)
                    features['pitch_min'] = np.zeros(12)
                    features['pitch_max'] = np.zeros(12)
                    features['pitch_median'] = np.zeros(12)

                # Loudness features from segments
                segments_loudness_max = h5['analysis']['segments_loudness_max'][:]
                if len(segments_loudness_max) > 0:
                    features['loudness_max_mean'] = float(
                        np.mean(segments_loudness_max))
                    features['loudness_max_std'] = float(
                        np.std(segments_loudness_max))
                else:
                    features['loudness_max_mean'] = 0.0
                    features['loudness_max_std'] = 0.0

                # Additional segment features
                segments_start = h5['analysis']['segments_start'][:]
                if len(segments_start) > 1:
                    # Segment density (segments per second)
                    features['segment_density'] = len(
                        segments_start) / features['duration'] if features['duration'] > 0 else 0
                else:
                    features['segment_density'] = 0.0

                return features

        except Exception as e:
            logger.error(f"Error processing {h5_path}: {str(e)}")
            return None

    def extract_all_features(self, max_files: int = None) -> pd.DataFrame:
        """Extract features from all HDF5 files"""

        if not self.h5_files:
            self.find_h5_files(max_files)

        all_features = []

        logger.info(f"Extracting features from {len(self.h5_files)} files...")

        for h5_file in tqdm(self.h5_files, desc="Extracting features"):
            features = self.extract_features_from_h5(h5_file)
            if features:
                all_features.append(features)

        logger.info(
            f"Successfully extracted features from {len(all_features)} files")

        # Convert to DataFrame
        df = self._features_to_dataframe(all_features)

        return df

    def _features_to_dataframe(self, features_list: List[Dict]) -> pd.DataFrame:
        """Convert list of feature dictionaries to DataFrame"""

        rows = []
        for features in features_list:
            row = {
                'track_id': features['track_id'],
                'artist_name': features['artist_name'],
                'title': features['title'],
                'duration': features['duration'],
                'tempo': features['tempo'],
                'loudness': features['loudness'],
                'key': features['key'],
                'mode': features['mode'],
                'time_signature': features['time_signature'],
                'energy': features['energy'],
                'danceability': features['danceability'],
                'loudness_max_mean': features['loudness_max_mean'],
                'loudness_max_std': features['loudness_max_std'],
                'segment_density': features['segment_density'],
            }

            # Add timbre features
            for i in range(12):
                row[f'timbre_mean_{i}'] = features['timbre_mean'][i]
                row[f'timbre_std_{i}'] = features['timbre_std'][i]
                row[f'timbre_min_{i}'] = features['timbre_min'][i]
                row[f'timbre_max_{i}'] = features['timbre_max'][i]
                row[f'timbre_median_{i}'] = features['timbre_median'][i]

            # Add pitch features
            for i in range(12):
                row[f'pitch_mean_{i}'] = features['pitch_mean'][i]
                row[f'pitch_std_{i}'] = features['pitch_std'][i]
                row[f'pitch_min_{i}'] = features['pitch_min'][i]
                row[f'pitch_max_{i}'] = features['pitch_max'][i]
                row[f'pitch_median_{i}'] = features['pitch_median'][i]

            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of numeric feature columns (excluding metadata)"""
        metadata_cols = ['track_id', 'artist_name', 'title']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        return feature_cols


def main():
    """Example usage"""
    from config import DATA_DIR, OUTPUT_DIR

    extractor = MillionSongFeatureExtractor(DATA_DIR)

    # Extract features (limit to 1000 files for testing, set to None for all)
    df = extractor.extract_all_features(max_files=None)

    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, 'extracted_features.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Features saved to {output_path}")

    # Print summary
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nFeature columns: {extractor.get_feature_columns(df)}")
    print(f"\nBasic statistics:")
    print(df.describe())


if __name__ == "__main__":
    main()
