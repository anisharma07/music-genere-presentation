"""
Feature Extraction Module
Extracts audio features from uploaded audio files using librosa
Features extracted match those in the GTZAN dataset
"""

import librosa
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def extract_features(audio_path, duration=30):
    """
    Extract audio features from a file (first 30 seconds)

    Parameters:
    -----------
    audio_path : str
        Path to the audio file
    duration : int
        Duration in seconds to analyze (default: 30)

    Returns:
    --------
    dict : Dictionary containing all extracted features
    """
    try:
        # Load audio file (first 30 seconds)
        y, sr = librosa.load(audio_path, duration=duration)

        features = {}

        # Length
        features['length'] = int(len(y))

        # Chroma STFT
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_stft_mean'] = float(np.mean(chroma_stft))
        features['chroma_stft_var'] = float(np.var(chroma_stft))

        # RMS (Root Mean Square Energy)
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = float(np.mean(rms))
        features['rms_var'] = float(np.var(rms))

        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_centroid_var'] = float(np.var(spectral_centroid))

        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = float(
            np.mean(spectral_bandwidth))
        features['spectral_bandwidth_var'] = float(np.var(spectral_bandwidth))

        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['rolloff_mean'] = float(np.mean(rolloff))
        features['rolloff_var'] = float(np.var(rolloff))

        # Zero Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate_mean'] = float(
            np.mean(zero_crossing_rate))
        features['zero_crossing_rate_var'] = float(np.var(zero_crossing_rate))

        # Harmonic and Percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        features['harmony_mean'] = float(np.mean(y_harmonic))
        features['harmony_var'] = float(np.var(y_harmonic))
        features['perceptr_mean'] = float(np.mean(y_percussive))
        features['perceptr_var'] = float(np.var(y_percussive))

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo) if isinstance(
            tempo, (np.ndarray, np.generic)) else tempo

        # MFCCs (Mel-frequency cepstral coefficients) - 20 coefficients
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(1, 21):
            features[f'mfcc{i}_mean'] = float(np.mean(mfccs[i-1]))
            features[f'mfcc{i}_var'] = float(np.var(mfccs[i-1]))

        return features

    except Exception as e:
        raise Exception(f"Error extracting features: {str(e)}")


def features_to_array(features_dict):
    """
    Convert features dictionary to numpy array in the correct order

    Parameters:
    -----------
    features_dict : dict
        Dictionary containing extracted features

    Returns:
    --------
    numpy.ndarray : Array of features in correct order
    """
    # Define the correct order of features (excluding filename and label)
    feature_names = [
        'length', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
        'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean',
        'spectral_bandwidth_var', 'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean',
        'zero_crossing_rate_var', 'harmony_mean', 'harmony_var', 'perceptr_mean',
        'perceptr_var', 'tempo', 'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var',
        'mfcc3_mean', 'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var',
        'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var',
        'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean', 'mfcc11_var',
        'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var',
        'mfcc15_mean', 'mfcc15_var', 'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var',
        'mfcc18_mean', 'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var'
    ]

    # Create array in correct order
    feature_array = np.array([float(features_dict[name])
                             for name in feature_names], dtype=np.float64)
    return feature_array.reshape(1, -1)


if __name__ == "__main__":
    # Test the feature extraction
    print("Feature Extraction Module")
    print("This module extracts audio features from audio files.")
    print("\nFeatures extracted:")
    print("  - Chroma STFT (mean, variance)")
    print("  - RMS Energy (mean, variance)")
    print("  - Spectral Centroid (mean, variance)")
    print("  - Spectral Bandwidth (mean, variance)")
    print("  - Spectral Rolloff (mean, variance)")
    print("  - Zero Crossing Rate (mean, variance)")
    print("  - Harmony (mean, variance)")
    print("  - Perceptr (mean, variance)")
    print("  - Tempo")
    print("  - MFCCs 1-20 (mean, variance)")
    print("\nTotal features: 58")
